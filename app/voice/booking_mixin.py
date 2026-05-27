"""
Booking & Calendar Token Mixin for BidirectionalStreamHandler.
Handles calendar slot caching, booking intent detection, and appointment token processing.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, date, timezone
from typing import TYPE_CHECKING, List, Optional

from app.core.config import settings
from app.core.logger import logger
from app.models.appointment import Appointment
from app.services.calendar_service import calendar_service
from app.utils.eleven_tts_text import strip_eleven_v3_style_tags_for_non_eleven_tts

if TYPE_CHECKING:
    pass

_RE_VOICE_END_CALL = re.compile(r"\[\s*END_CALL\s*\]", re.IGNORECASE)
_RE_VOICE_SCREENING_QUALIFIED = re.compile(r"\[\s*SCREENING_QUALIFIED\s*\]", re.IGNORECASE)


class BookingMixin:
    """Calendar and appointment booking methods for BidirectionalStreamHandler."""

    # ── Calendar token handlers ───────────────────────────────────────────────

    @staticmethod
    def _normalize_calendar_slot_key(value: str) -> str:
        value = (value or "").strip().lower().replace(".", "")
        return re.sub(r"\s+", " ", value)

    def _cache_calendar_slots(self, slots: List) -> None:
        self._last_offered_calendar_slots = [slot.slot_start for slot in slots]
        self._last_selected_calendar_slot = None

    @staticmethod
    def _has_calendar_token(text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"\[\s*(?:CHECK_SLOTS|BOOK_APPOINTMENT)\s*:", text, flags=re.IGNORECASE))

    def _is_booking_intent_turn(self, user_text: str, model_text: str = "") -> bool:
        """Conservative booking-intent detector for token-budget and fallback extraction."""
        haystack = f"{user_text or ''} {model_text or ''}".lower()
        if not haystack.strip():
            return False
        booking_keywords = (
            "book", "booking", "schedule", "appointment", "reschedule", "slot", "available slot",
            "am", "pm", "a.m", "p.m", "date", "time", "tomorrow", "today",
        )
        return any(k in haystack for k in booking_keywords)

    def _should_use_latency_fastpath(self, user_text: str, booking_intent_turn: bool) -> bool:
        """
        Fast-path only for short/simple turns where heavy context is unlikely to help.
        Keeps booking/business-intent turns on the full context path.
        """
        if not bool(getattr(settings, "VOICE_ENABLE_LATENCY_FASTPATH", True)):
            return False
        if booking_intent_turn:
            return False
        # Never use fast path when a restricted service area is configured: any turn
        # could carry a location statement ("Houston, Texas" = 2 words) and needs
        # the full KB + policy block so the service area gate fires correctly.
        if (
            self._kb_cache_ready
            and self._cached_business_knowledge_block
            and "COVERAGE: RESTRICTED" in self._cached_business_knowledge_block
        ):
            return False
        text = (user_text or "").strip().lower()
        if not text:
            return False
        max_words = int(getattr(settings, "VOICE_FASTPATH_MAX_WORDS", 7) or 7)
        words = text.split()
        if len(words) > max_words:
            return False
        heavy_intent_markers = (
            "price", "pricing", "cost", "address", "phone", "email", "website",
            "service", "book", "booking", "appointment", "schedule", "slot", "quote",
            "area", "location", "city", "state", "zip", "county", "region",
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
            if (now - ts) < self._DUP_USER_TURN_WINDOW_SEC and u_norm and u_norm == user_norm:
                return True
        return False

    def _is_duplicate_agent_line(self, user_text: Optional[str], agent_text: str) -> bool:
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
            if len(a_norm) > 30 and (a_norm.startswith(prev_a) or prev_a.startswith(a_norm)):
                shorter, longer = sorted((a_norm, prev_a), key=len)
                if shorter and len(shorter) / max(len(longer), 1) >= 0.9:
                    # And the user turn matches (or was empty) — safe to dedupe.
                    if not u_norm or not prev_u or prev_u == u_norm:
                        return True
        return False

    _INFLIGHT_TTS_ECHO_WINDOW_SEC: float = 15.0
    _INFLIGHT_TTS_SNIPPETS_MAX: int = 12

    def _clear_inflight_tts_echo_guard(self) -> None:
        """Reset per-turn TTS snippets used for interim barge-in echo detection."""
        self._inflight_tts_snippets: list[tuple[str, float]] = []

    def _record_inflight_tts_for_echo_guard(self, text: str) -> None:
        """
        Remember text queued/spoken this turn so interim STT echo does not barge-in.

        Unlike ``_recent_agent_pairs`` (filled after transcript commit), this tracks
        partial TTS flushes while the agent is still speaking.
        """
        snippet = (text or "").strip()
        if not snippet:
            return
        norm = self._normalize_turn_text(snippet)
        if not norm:
            return
        if not hasattr(self, "_inflight_tts_snippets"):
            self._inflight_tts_snippets = []
        now = time.monotonic()
        self._inflight_tts_snippets.append((norm, now))
        cutoff = now - self._INFLIGHT_TTS_ECHO_WINDOW_SEC
        self._inflight_tts_snippets = [
            (n, ts) for n, ts in self._inflight_tts_snippets if ts >= cutoff
        ][-self._INFLIGHT_TTS_SNIPPETS_MAX :]

    def _stt_overlaps_spoken_text(self, transcript: str, spoken_norm: str, *, min_words: int) -> bool:
        t_norm = self._normalize_turn_text(transcript)
        if not t_norm or not spoken_norm:
            return False
        t_words = t_norm.split()
        if len(t_words) < min_words:
            return False
        if len(t_words) >= 2 and (t_norm in spoken_norm or spoken_norm in t_norm):
            return True
        t_set = set(t_words)
        s_set = set(spoken_norm.split())
        if len(s_set) < min_words:
            return False
        overlap = len(t_set & s_set)
        shorter = min(len(t_set), len(s_set))
        return shorter > 0 and overlap / shorter >= 0.80

    def _is_agent_self_echo(self, transcript: str, *, min_words: int = 4) -> bool:
        """
        True when incoming STT text closely matches text the agent recently spoke.

        Phone-line sidetone and open-mic setups can feed the agent's TTS audio back
        into the STT stream.  Finals use ``min_words=4``; interim barge-in uses
        ``_is_likely_agent_echo_for_barge_in`` (min_words=2) so short echoes like
        "hello yes" do not cancel the rest of the reply.

        Window: 12 s — covers TTS latency, full playback, and STT pipeline lag.
        """
        if not transcript:
            return False
        now = time.monotonic()
        for _u, a_norm, ts in getattr(self, "_recent_agent_pairs", []):
            if (now - ts) > 12.0 or not a_norm:
                continue
            if self._stt_overlaps_spoken_text(transcript, a_norm, min_words=min_words):
                return True
        return False

    def _is_likely_agent_echo_for_barge_in(self, transcript: str) -> bool:
        """Interim barge-in guard: suppress cancel when STT likely picked up agent TTS."""
        if not transcript:
            return False
        now = time.monotonic()
        for spoken_norm, ts in getattr(self, "_inflight_tts_snippets", []):
            if (now - ts) > self._INFLIGHT_TTS_ECHO_WINDOW_SEC:
                continue
            if self._stt_overlaps_spoken_text(transcript, spoken_norm, min_words=2):
                return True
        return self._is_agent_self_echo(transcript, min_words=2)

    def _extract_caller_location_from_transcript(self) -> str:
        """
        Scan the in-memory conversation history for a client message that looks like a
        location response.  Returns the first matching client utterance (truncated to 80
        chars) or an empty string when nothing is found.

        Heuristic: a client message is treated as a location if it contains a US state
        name/abbreviation, or if it was immediately preceded by an agent turn that asked
        about city/state/location.  Scanning is limited to the most recent 30 pairs so
        the check stays O(1) on long calls.
        """
        import re as _re

        # Common US state names and two-letter abbreviations (covers the vast majority of
        # service-area lookups; non-US businesses will typically use global coverage).
        _STATE_RE = _re.compile(
            r"\b(?:alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|"
            r"florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|"
            r"maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|"
            r"nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|"
            r"north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|"
            r"south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|"
            r"wisconsin|wyoming|"
            r"\bAL\b|\bAK\b|\bAZ\b|\bAR\b|\bCA\b|\bCO\b|\bCT\b|\bDE\b|\bFL\b|\bGA\b|"
            r"\bHI\b|\bID\b|\bIL\b|\bIN\b|\bIA\b|\bKS\b|\bKY\b|\bLA\b|\bME\b|\bMD\b|"
            r"\bMA\b|\bMI\b|\bMN\b|\bMS\b|\bMO\b|\bMT\b|\bNE\b|\bNV\b|\bNH\b|\bNJ\b|"
            r"\bNM\b|\bNY\b|\bNC\b|\bND\b|\bOH\b|\bOK\b|\bOR\b|\bPA\b|\bRI\b|\bSC\b|"
            r"\bSD\b|\bTN\b|\bTX\b|\bUT\b|\bVT\b|\bVA\b|\bWA\b|\bWV\b|\bWI\b|\bWY\b)",
            _re.IGNORECASE,
        )
        _LOCATION_QUESTION_RE = _re.compile(
            r"\b(?:city|state|zip|area|location|address|where|neighbourhood|neighborhood|borough|county|region)\b",
            _re.IGNORECASE,
        )

        history = self._conversation_history_cache[-60:]  # last 30 pairs max
        prev_agent_asked_location = False
        for role, text in history:
            if role == "agent":
                prev_agent_asked_location = bool(_LOCATION_QUESTION_RE.search(text or ""))
            elif role in ("client", "user"):
                txt = (text or "").strip()
                if not txt:
                    prev_agent_asked_location = False
                    continue
                if _STATE_RE.search(txt) or prev_agent_asked_location:
                    return txt[:80]
                prev_agent_asked_location = False
        return ""

    def _is_location_in_service_area(self, location: str) -> bool:
        """
        Return True when *location* appears to be within the business's service areas.

        Reads the verbatim service areas text from _cached_business_knowledge_block.
        Uses word-level intersection: if any significant word from the caller's location
        (3+ chars, not a stop-word) appears in the service areas blob, we treat it as
        a match.  This is intentionally permissive — a false-negative (blocking a valid
        caller) is far worse than a false-positive here; the LLM gate handles nuance.
        """
        if not location or not self._cached_business_knowledge_block:
            return True  # no block to check against — allow by default

        service_area_text = self._get_service_area_text_from_bk_block()
        if not service_area_text:
            return True  # service areas not parseable — allow

        loc_lower = location.lower()
        sa_lower = service_area_text.lower()

        # Direct substring match first (fast path for well-formed data).
        if loc_lower in sa_lower:
            return True

        _stop = {"in", "the", "of", "and", "or", "at", "to", "for", "a", "is", "are", "we", "our"}
        loc_words = [w for w in loc_lower.split() if len(w) >= 3 and w not in _stop]
        if not loc_words:
            return True  # location too vague to deny — allow

        return any(word in sa_lower for word in loc_words)

    def _get_service_area_text_from_bk_block(self) -> str:
        """
        Extract the verbatim 'Service Areas (verbatim): ...' line from the cached BK block.
        Returns the area text, or an empty string when the line is not present.
        """
        import re as _re

        block = self._cached_business_knowledge_block or ""
        m = _re.search(r"Service Areas \(verbatim\):\s*(.+)", block)
        return m.group(1).strip() if m else ""

    def _remember_agent_turn(self, user_text: Optional[str], agent_text: str) -> None:
        """Append (user_norm, agent_norm, ts) and bound the buffer to the last few entries."""
        if not agent_text:
            return
        a_norm = self._normalize_turn_text(agent_text)
        if not a_norm:
            return
        u_norm = self._normalize_turn_text(user_text or "")
        self._recent_agent_pairs.append((u_norm, a_norm, time.monotonic()))
        if len(self._recent_agent_pairs) > self._RECENT_AGENT_PAIRS_MAX:
            self._recent_agent_pairs = self._recent_agent_pairs[-self._RECENT_AGENT_PAIRS_MAX :]

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
            seed_slot = self._resolve_cached_calendar_slot(self._turn_response_seed_text)
            if final_slot and seed_slot and final_slot != seed_slot:
                return True
            return False

        # Booking: slot/date resolution changed — re-run
        if self._is_booking_context_active(final_transcript) or self._is_booking_context_active(
            self._turn_response_seed_text
        ):
            final_slot = self._resolve_cached_calendar_slot(final_transcript)
            seed_slot = self._resolve_cached_calendar_slot(self._turn_response_seed_text)
            if final_slot != seed_slot:
                return True
            correction_markers = ("wrong", "no no", "not ", "already", "spell", "11 00", "11 am")
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

    def _build_follow_up_appointment_block(self) -> str:
        """Extra instructions when this outbound call was scheduled as an appointment reminder."""
        if not self.call_session or not self.call_session.call_metadata:
            return ""
        aid_raw = str(self.call_session.call_metadata.get("appointment_id") or "").strip()
        if not aid_raw:
            return ""
        appt = None
        try:
            appt_id = uuid.UUID(aid_raw)
            appt = (
                self.db.query(Appointment)
                .filter(
                    Appointment.id == appt_id,
                    Appointment.tenant_id == self.call_session.tenant_id,
                )
                .first()
            )
        except Exception:
            appt = None

        details: list[str] = []
        if appt:
            if (appt.customer_name or "").strip():
                details.append(f"- Customer name: {appt.customer_name.strip()}.")
            if (appt.customer_phone or "").strip():
                details.append(f"- Customer phone: {appt.customer_phone.strip()}.")
            if (appt.appointment_reason or "").strip():
                details.append(f"- Appointment reason: {appt.appointment_reason.strip()}.")
            if appt.slot_start:
                try:
                    from app.services.calendar_service import calendar_service as _calendar_service

                    _tz_label, slot_start_local, _slot_end_local = _calendar_service.appointment_local_display(
                        self.db,
                        self.call_session.tenant_id,
                        appt,
                    )
                    details.append(
                        "- Scheduled appointment time (local): "
                        f"{slot_start_local.strftime('%A, %B %d at %I:%M %p').replace(' 0', ' ')}."
                    )
                except Exception:
                    pass
        else:
            details.append(f"- Appointment ID: {aid_raw}.")

        details_block = "\n".join(details)
        if details_block:
            details_block = f"{details_block}\n"

        return (
            "# APPOINTMENT FOLLOW-UP REMINDER (THIS CALL ONLY)\n"
            f"{details_block}"
            "- The customer has an appointment on file. Confirm whether they (or someone for the service) will attend at the scheduled time.\n"
            "- When you mention the appointment time, use the local time shown above. Do not mention UTC or any timezone name.\n"
            "- If they clearly confirm attendance: thank them briefly, then put [FOLLOWUP_CONFIRM] alone on its own line at the end of your message, "
            "then end with [END_CALL] on the next line.\n"
            "- If they want to cancel the appointment: acknowledge, then put [FOLLOWUP_CANCEL] alone on its own line at the end.\n"
            "- If they want to reschedule: collect a concrete new date and time they agree to, then put exactly one line at the end: "
            "[FOLLOWUP_RESCHEDULE:slot=<ISO8601 datetime>]. Use UTC with offset or Z (e.g. 2026-05-10T15:00:00+00:00). "
            "If they have not given a new time yet, ask for it — do not emit RESCHEDULE until the slot is explicit.\n"
            "- Do not use [BOOK_APPOINTMENT:...] in this reminder flow.\n"
        )

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
        during the call — the actual Appointment row is created post-call by
        post_call_appointment_service after contact + slot validation. If the
        LLM ignores that rule and says "your appointment is booked",
        "your plumbing service is booked", "you're all set", etc., we strip
        those sentences before they reach TTS so the caller never hears a
        false confirmation.
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
        out = re.sub(r"\[CHECK_SLOTS:[^\]]*\]", "", out)
        out = re.sub(r"\[BOOK_APPOINTMENT:[^\]]*\]", "", out)
        out = re.sub(r"\[\s*FOLLOWUP_CONFIRM\s*\]", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\[\s*FOLLOWUP_CANCEL\s*\]", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\[\s*FOLLOWUP_RESCHEDULE:[^\]]*\]", "", out, flags=re.IGNORECASE)
        # Strip all known ElevenLabs-style audio tags and common variants so they
        # are never spoken as literal words regardless of TTS provider.
        out = strip_eleven_v3_style_tags_for_non_eleven_tts(out)
        # Citation-like bracket numbers such as [1], [2], ...
        out = re.sub(r"\[\s*\d{1,3}\s*\]", "", out)
        # Malformed bracket-open tokens without closing bracket
        out = re.sub(
            r"\[(?:OUTCOME|CHECK_SLOTS|BOOK_APPOINTMENT|FOLLOWUP_RESCHEDULE):[^\]\n\r]*",
            "",
            out,
        )
        # Unbracketed control tails occasionally produced by model
        out = re.sub(
            r"(?im)\b(?:OUTCOME|CHECK_SLOTS|BOOK_APPOINTMENT|FOLLOWUP_RESCHEDULE)\s*:\s*[^\n\r]*",
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
            r"\bbook_appointment\b",
            r"\bcheck_slots\b",
            r"\boutcome\b",
            r"\bslot\s*=",
            r"\breason\s*=",
            r"\bname\s*=",
            r"\bemail\s*=",
            r"\bphone\s*=",
            r"\bclient phone number slot\b",
        )
        return any(re.search(p, t, flags=re.IGNORECASE) for p in leak_patterns)

    def _prepare_tts_text(self, text: str) -> str:
        """
        Final text gate before queueing TTS.
        """
        cleaned = self._strip_control_tokens_for_tts(text or "")
        cleaned = self._strip_premature_booking_confirmation(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if self._looks_like_control_leak(cleaned):
            logger.warning("TTSGuard: dropped token-like leak text=%r", cleaned[:180])
            return ""
        return cleaned

    async def _extract_calendar_action_token(
        self,
        *,
        llm_service,
        model_name: str,
        api_key: Optional[str],
        user_text: str,
        assistant_text: str,
        history_text: str,
        temperature: float,
    ) -> Optional[str]:
        """Second-pass action extraction. Returns one action token or None."""
        if not self.call_session:
            return None

        offered_slots = ", ".join(
            slot.strftime("%Y-%m-%d %H:%M")
            for slot in self._last_offered_calendar_slots[:16]
        )
        extraction_system_prompt = (
            "You suggest a single calendar hint line from a phone-call turn. "
            "Output is not authoritative; the server validates everything.\n"
            "Return exactly one line and nothing else:\n"
            "- [BOOK_APPOINTMENT:name=<placeholder>,slot=<slot>,reason=<reason>] "
            "(optional phone=...,email=...)\n"
            "- [CHECK_SLOTS:date=YYYY-MM-DD]\n"
            "- NONE\n"
            "Rules:\n"
            "1) If user selected a concrete offered slot, return BOOK_APPOINTMENT with slot.\n"
            "2) If user asked to check availability, return CHECK_SLOTS.\n"
            "3) If uncertain, return NONE.\n"
            "4) Keep reason short and without commas.\n"
        )
        extraction_prompt = (
            f"Now (UTC): {datetime.now(timezone.utc).isoformat()}\n\n"
            f"Recent history:\n{history_text or '(empty)'}\n\n"
            f"Latest user text:\n{user_text or '(empty)'}\n\n"
            f"Assistant draft text:\n{assistant_text or '(empty)'}\n\n"
            f"Offered slot starts (YYYY-MM-DD HH:MM):\n{offered_slots or '(none cached)'}\n"
        )

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: llm_service.generate_text(
                    prompt=extraction_prompt,
                    system_prompt=extraction_system_prompt,
                    model_name=model_name,
                    temperature=min(temperature, 0.15),
                    max_tokens=180,
                    api_key=api_key,
                ),
            )
            content = (result.get("content") or "").strip()
            if re.search(r"^\[\s*BOOK_APPOINTMENT\s*:", content, flags=re.IGNORECASE):
                return content.splitlines()[0].strip()
            if re.search(r"^\[\s*CHECK_SLOTS\s*:", content, flags=re.IGNORECASE):
                return content.splitlines()[0].strip()
            return None
        except Exception as e:
            logger.warning("Calendar action extraction pass failed: %s", e)
            return None

    def _resolve_cached_calendar_slot(self, slot_raw: str) -> Optional[datetime]:
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
                self._normalize_calendar_slot_key(candidate)
                for candidate in candidates
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
                    offered_local = slot_dt.replace(tzinfo=None, second=0, microsecond=0)
                    parsed_local = parsed_dt.replace(second=0, microsecond=0)
                    if offered_local == parsed_local:
                        return slot_dt
                else:
                    offered_utc = slot_dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
                    parsed_utc = parsed_dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
                    if offered_utc == parsed_utc:
                        return slot_dt

        for fmt in ("%I:%M %p", "%I %p", "%H:%M"):
            try:
                parsed_time = datetime.strptime(slot_raw.strip(), fmt).time()
                matches = [
                    slot_dt
                    for slot_dt in self._last_offered_calendar_slots
                    if slot_dt.hour == parsed_time.hour and slot_dt.minute == parsed_time.minute
                ]
                if len(matches) == 1:
                    return matches[0]
            except ValueError:
                continue

        return None

    async def _handle_check_slots_token(self, llm_response: str):
        """
        Called when LLM emits [CHECK_SLOTS:date=<value>].
        Fetches available slots and speaks them directly via TTS (no second LLM call).
        """
        try:
            import re as _re
            from datetime import date as _date, timedelta as _td, datetime as _dt, timezone as _tz
            from zoneinfo import ZoneInfo as _ZI
            from app.services.calendar_service import calendar_service as _cal

            m = _re.search(r"\[CHECK_SLOTS:date=([^\]]+)\]", llm_response)
            if not m:
                return

            if not self.call_session:
                return

            tenant_id = self.call_session.tenant_id
            agent_id = self.agent.id if self.agent else None

            loop = asyncio.get_running_loop()
            tenant_tz_str = await loop.run_in_executor(
                None,
                lambda: _cal.get_tenant_timezone(self.db, tenant_id),
            )
            try:
                tenant_tz = _ZI(tenant_tz_str)
            except Exception:
                tenant_tz = _tz.utc

            today = _dt.now(tenant_tz).date()
            raw_date = m.group(1).strip().lower()

            if raw_date in ("today", "aaj"):
                target = today
            elif raw_date in ("tomorrow", "kal", "tomorrow's"):
                target = today + _td(days=1)
            else:
                try:
                    target = _date.fromisoformat(raw_date)
                except ValueError:
                    target = today + _td(days=1)

            result = await loop.run_in_executor(
                None,
                lambda: _cal.get_available_slots(self.db, tenant_id, target, agent_id),
            )
            self._cache_calendar_slots(result.slots)
            self._last_requested_calendar_date = target

            if not result.slots:
                msg = f"Sorry, there are no available slots on {target.strftime('%A, %B %d')}. Please try another date."
            else:
                slot_labels = ", ".join(s.slot_label for s in result.slots[:6])
                suffix = f" and {len(result.slots) - 6} more" if len(result.slots) > 6 else ""
                msg = (
                    f"On {target.strftime('%A, %B %d')}, these slots are available: "
                    f"{slot_labels}{suffix}. Which time works for you?"
                )

            await self._add_to_transcript("agent", msg, "calendar_slots")
            if self._tts_pipeline:
                await self._tts_pipeline.queue_tts({
                    "text": msg,
                    "chunk_id": "calendar_slots",
                    "use_ssml": False,
                    "is_final": True,
                })
        except Exception as e:
            logger.error("Error in _handle_check_slots_token: %s", e, exc_info=True)

    def _client_transcript_lines_newest_first(self, limit: int = 16) -> list[str]:
        """Recent client utterances (newest first) for voice email recovery."""
        conversation_history: list = []
        if self.call_session and self.call_session.call_transcript:
            try:
                raw = self.call_session.call_transcript
                conversation_history = (
                    json.loads(raw) if isinstance(raw, str) else raw
                )
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

    def _follow_up_appointment_uuid(self) -> Optional[uuid.UUID]:
        if not self.call_session or not self.call_session.call_metadata:
            return None
        raw = (self.call_session.call_metadata.get("appointment_id") or "").strip()
        if not raw:
            return None
        try:
            return uuid.UUID(raw)
        except ValueError:
            return None

    async def _handle_followup_confirm_token(self, llm_response: str) -> None:
        try:
            appt_id = self._follow_up_appointment_uuid()
            if not appt_id or not self.call_session:
                return
            from app.services.appointment_follow_up_service import send_follow_up_outcome_staff_email

            send_follow_up_outcome_staff_email(
                self.db,
                staff_user_id=self.call_session.user_id,
                tenant_id=self.call_session.tenant_id,
                appointment_id=appt_id,
                outcome="confirmed_attendance",
                detail="Customer confirmed attendance on the reminder call.",
            )
        except Exception as e:
            logger.error("Error in _handle_followup_confirm_token: %s", e, exc_info=True)

    async def _handle_followup_cancel_token(self, llm_response: str) -> None:
        try:
            appt_id = self._follow_up_appointment_uuid()
            if not appt_id or not self.call_session:
                return
            from app.services.calendar_service import calendar_service
            from app.services.appointment_follow_up_service import send_follow_up_outcome_staff_email

            calendar_service.update_appointment_status(
                self.db,
                appt_id,
                self.call_session.tenant_id,
                "cancelled",
                cancellation_reason="Customer requested cancellation on reminder call.",
                notify_user_id=self.call_session.user_id,
            )
            send_follow_up_outcome_staff_email(
                self.db,
                staff_user_id=self.call_session.user_id,
                tenant_id=self.call_session.tenant_id,
                appointment_id=appt_id,
                outcome="cancelled",
                detail="Appointment cancelled from reminder call.",
            )
            try:
                self.db.refresh(self.call_session)
            except Exception:
                pass
        except ValueError as ve:
            logger.warning("Follow-up cancel: %s", ve)
        except Exception as e:
            logger.error("Error in _handle_followup_cancel_token: %s", e, exc_info=True)

    async def _handle_followup_reschedule_token(self, llm_response: str) -> None:
        try:
            appt_id = self._follow_up_appointment_uuid()
            if not appt_id or not self.call_session:
                return
            m = re.search(r"\[\s*FOLLOWUP_RESCHEDULE:([^\]]+)\]", llm_response, flags=re.IGNORECASE)
            if not m:
                return
            inner = " ".join((m.group(1) or "").split())
            sm = re.search(r"slot=(?P<slot>.+)", inner, flags=re.IGNORECASE)
            slot_raw = (sm.group("slot") or "").strip() if sm else ""
            if not slot_raw:
                logger.warning("FOLLOWUP_RESCHEDULE missing slot: %s", inner[:300])
                return
            from datetime import datetime as _dt

            from app.services.calendar_service import calendar_service
            from app.services.appointment_follow_up_service import send_follow_up_outcome_staff_email

            try:
                slot_start = _dt.fromisoformat(slot_raw.replace("Z", "+00:00"))
            except ValueError:
                resolved = self._resolve_cached_calendar_slot(slot_raw)
                if resolved is None:
                    logger.warning("FOLLOWUP_RESCHEDULE invalid slot: %s", slot_raw)
                    return
                slot_start = resolved

            calendar_service.reschedule_appointment(
                db=self.db,
                tenant_id=self.call_session.tenant_id,
                appointment_id=appt_id,
                slot_start=slot_start,
                notify_user_id=self.call_session.user_id,
            )
            send_follow_up_outcome_staff_email(
                self.db,
                staff_user_id=self.call_session.user_id,
                tenant_id=self.call_session.tenant_id,
                appointment_id=appt_id,
                outcome="rescheduled",
                detail=f"New slot (UTC instant): {slot_start.isoformat()}",
            )
            try:
                self.db.refresh(self.call_session)
            except Exception:
                pass
        except ValueError as ve:
            logger.warning("Follow-up reschedule: %s", ve)
        except Exception as e:
            logger.error("Error in _handle_followup_reschedule_token: %s", e, exc_info=True)

    async def _handle_book_appointment_token(self, llm_response: str):
        """
        LLM may emit [BOOK_APPOINTMENT:...] as a non-authoritative intent hint.
        Backend stores only slot / reason in call_metadata.booking_intent.
        Name and email from the token are ignored. No in-call reservation or appointment commit.
        Final booking runs in post_call_appointment_service after validation.

        Service area gate: if COVERAGE: RESTRICTED, the caller's location must be present
        (from the token's location= field or from the transcript) and confirmed within the
        listed service areas before the booking intent is stored.
        """
        try:
            import re as _re
            from datetime import datetime as _dt

            from app.services.call_session_contact_state import persist_booking_intent_fields

            m = _re.search(r"\[BOOK_APPOINTMENT:([^\]]+)\]", llm_response)
            if m:
                raw = m.group(1)
            else:
                # Tolerate malformed token without closing bracket during live calls.
                m_fallback = _re.search(r"\[BOOK_APPOINTMENT:(.+)$", llm_response, flags=_re.DOTALL)
                if not m_fallback:
                    return
                raw = m_fallback.group(1).strip()
                logger.warning(
                    "BOOK_APPOINTMENT token missing closing bracket; using fallback parser. token_tail=%s",
                    raw[:300],
                )

            raw_single_line = " ".join((raw or "").split())

            # Backward-compatible key extractor (used for location and fallback slot/reason).
            def _get(key: str) -> str:
                km = _re.search(rf"{key}=([^,\]]+)", raw_single_line)
                return km.group(1).strip() if km else ""

            # Robust parse: name, optional location, optional phone/email, slot, optional reason.
            strict = _re.search(
                r"name=(?P<name>.*?),\s*(?:location=(?P<location>.*?),\s*)?"
                r"(?:phone=(?P<phone>.*?),\s*)?(?:email=(?P<email>.*?),\s*)?"
                r"slot=(?P<slot>.*?)(?:,\s*reason=(?P<reason>.*))?$",
                raw_single_line,
            )
            if strict:
                slot_raw = (strict.group("slot") or "").strip()
                reason_val = (strict.group("reason") or "").strip()
                reason = reason_val or None
                location_from_token = (strict.group("location") or "").strip()
            else:
                slot_raw = _get("slot")
                reason = _get("reason") or None
                location_from_token = _get("location")

            if not slot_raw:
                logger.warning("BOOK_APPOINTMENT token missing slot: %s", raw_single_line[:500])
                return

            if not self.call_session:
                return

            # --- Service Area Gate ---
            # If the business has restricted coverage, validate caller location before
            # storing any booking intent. This is the last line of defence — the LLM
            # prompt already instructs the model not to emit BOOK_APPOINTMENT without a
            # confirmed in-area location, but LLMs can bypass prompt rules.
            if (
                self._cached_business_knowledge_block
                and "COVERAGE: RESTRICTED" in self._cached_business_knowledge_block
            ):
                caller_location = location_from_token or self._extract_caller_location_from_transcript()

                if not caller_location:
                    logger.warning(
                        "BOOK_APPOINTMENT blocked: COVERAGE: RESTRICTED but no caller location found. token=%s",
                        raw_single_line[:300],
                    )
                    msg = (
                        "Before I can schedule, I need to confirm your location. "
                        "What city and state is the property in?"
                    )
                    await self._add_to_transcript("agent", msg, "service_area_location_request")
                    if self._tts_pipeline:
                        await self._tts_pipeline.queue_tts({
                            "text": msg,
                            "chunk_id": "service_area_location_request",
                            "use_ssml": False,
                            "is_final": True,
                        })
                    return

                if not self._is_location_in_service_area(caller_location):
                    service_area_text = self._get_service_area_text_from_bk_block()
                    area_phrase = f" We serve {service_area_text}." if service_area_text else ""
                    logger.warning(
                        "BOOK_APPOINTMENT blocked: location=%r not in service area. token=%s",
                        caller_location,
                        raw_single_line[:300],
                    )
                    msg = (
                        f"I'm sorry, but we don't currently provide services in {caller_location}.{area_phrase} "
                        "Thank you for calling, and I hope you find the help you need."
                    )
                    await self._add_to_transcript("agent", msg, "service_area_rejection")
                    if self._tts_pipeline:
                        await self._tts_pipeline.queue_tts({
                            "text": msg,
                            "chunk_id": "service_area_rejection",
                            "use_ssml": False,
                            "is_final": True,
                        })
                    return

            slot_start = self._resolve_cached_calendar_slot(slot_raw)
            if slot_start is None:
                try:
                    slot_start = _dt.fromisoformat(slot_raw.replace("Z", "+00:00"))
                except ValueError:
                    logger.warning("BOOK_APPOINTMENT: invalid slot datetime: %s", slot_raw)
                    return

            slot_iso = slot_start.isoformat()

            persist_booking_intent_fields(
                self.db,
                self.call_session,
                slot_start_iso=slot_iso,
                appointment_reason=reason,
            )
            self._last_selected_calendar_slot = slot_start
            try:
                self.db.refresh(self.call_session)
            except Exception:
                pass

            msg = (
                "I've noted your preferred time. After we finish the call, our system will finalize "
                "your appointment if everything checks out. Anything else I can help with?"
            )

            await self._add_to_transcript("agent", msg, "calendar_booking")
            if self._tts_pipeline:
                await self._tts_pipeline.queue_tts({
                    "text": msg,
                    "chunk_id": "calendar_booking",
                    "use_ssml": False,
                    "is_final": True,
                })
        except Exception as e:
            logger.error("Error in _handle_book_appointment_token: %s", e, exc_info=True)

