import asyncio
import json
import random
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.core.logger import logger
from app.core.config import settings
from app.services.agent_service import agent_service
from app.services.openai_service import openai_service
from app.services.groq_service import groq_service
from app.utils.eleven_tts_text import (
    build_elevenlabs_audio_tag_prompt_block,
    get_elevenlabs_voice_prompt_rule_lines,
    strip_eleven_v3_style_tags_for_non_eleven_tts,
    supports_elevenlabs_audio_tags,
)


# ---------------------------------------------------------------------------
# Configuration structures (tunable parameters for STT, TTS, and conversation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuickAckConfig:
    """Config for quick acknowledgement behaviour."""

    min_words: int
    probability: float
    skip_phrases: Tuple[str, ...]


@dataclass(frozen=True)
class VoiceTunables:
    """High-level tunables for the bidirectional stream behaviour."""

    # STT → LLM trigger: as soon as we have ~this much STT stream, send to LLM
    stt_interim_interval_ms: int = settings.VOICE_STT_INTERIM_INTERVAL_MS

    # Conversation context: keep the prompt small for latency (voice calls)
    history_max_messages: int = settings.VOICE_HISTORY_MAX_MESSAGES

    # Incremental TTS: flush when we have a complete thought/sentence
    tts_flush_min_words: int = settings.VOICE_TTS_FLUSH_MIN_WORDS
    tts_flush_max_words: int = settings.VOICE_TTS_FLUSH_MAX_WORDS

    # Quick acknowledgement: 5-word rule + probability (Vapi+ naturalness)
    quick_ack: QuickAckConfig = QuickAckConfig(
        min_words=settings.VOICE_QUICK_ACK_MIN_WORDS,
        probability=settings.VOICE_QUICK_ACK_PROBABILITY,
        skip_phrases=(
            # Never say "Got it" to emotional/serious content
            "help",
            "emergency",
            "urgent",
            "problem",
            "issue",
            "sad",
            "angry",
            "please help",
            "asap",
            "critical",
            "wrong",
            "broken",
            "not working",
            "complaint",
        ),
    )


VOICE_TUNABLES = VoiceTunables()


# ---------------------------------------------------------------------------
# Small pure helpers (no side effects, easy to reason about)
# ---------------------------------------------------------------------------


def should_send_quick_ack(user_text: str, config: QuickAckConfig) -> bool:
    """
    Decide whether a quick acknowledgement is eligible for a given user text.

    This only answers the eligibility question (length / emotional filters),
    leaving probabilistic sampling to the caller.
    """
    text = (user_text or "").strip()
    if not text:
        return False

    words = text.split()
    if len(words) < config.min_words:
        return False

    lower = text.lower()
    for phrase in config.skip_phrases:
        if phrase in lower:
            return False

    return True


@dataclass
class ConversationActions:
    """
    High-level actions decided from a user speech event.
    The handler uses this to drive quick-acks, LLM responses, and history updates.
    """

    quick_ack_text: Optional[str] = None
    start_llm_response: bool = False
    end_call_after: bool = False

    # Updated conversation history (already windowed)
    updated_history: Optional[List[Dict[str, Any]]] = None
    should_persist_history: bool = False


class ConversationOrchestrator:
    """
    Encapsulates conversation + policy logic for a single bidirectional call:
    - Quick-ack rules (length/probability/banned phrases).
    - History windowing and prompt construction.
    - LLM provider/model selection and streaming.

    This class keeps a narrow dependency on the handler so that the WebSocket
    layer no longer needs to think about probabilities or thresholds.
    """

    def __init__(self, handler: Any):
        self._h = handler

    # ---- Interim processing / barge-in gating -------------------------

    async def process_interim(self, transcript: str, confidence: float) -> None:
        """
        Delegates to the bidirectional stream handler (single source of truth).
        """
        await self._h._maybe_process_interim(transcript, confidence)

    # ---- Quick acknowledgements ---------------------------------------

    async def send_quick_acknowledgement(self, user_text: str) -> None:
        """
        Send instant acknowledgement for longer queries while generating full response.
        Probability-based so we don't say "Got it" every time — more natural.
        Skips emotional/serious content so we never ack with "Got it" to e.g. "I have an emergency".
        """
        text = (user_text or "").strip()
        if not should_send_quick_ack(text, VOICE_TUNABLES.quick_ack):
            return

        # Apply probability filter so we don't say "Got it" every single time
        if random.random() >= VOICE_TUNABLES.quick_ack.probability:
            return

        acks = [
            "Got it",
            "I see",
            "Okay",
            "Alright",
            "Sure",
            "Mm-hmm",
            "Oh, okay",
            "One moment",
            "Hang on a sec",
            "Let me check that",
        ]
        ack = random.choice(acks)
        if not self._h._tts_pipeline:
            return
        await self._h._tts_pipeline.queue_tts(
            {
                "text": ack,
                "chunk_id": "quick_ack",
                "use_ssml": False,
                "is_acknowledgement": True,
                "is_final": False,
            }
        )

    # ---- LLM + history orchestration ----------------------------------

    async def generate_and_stream_response(
        self, user_text: str, confidence: float, is_greeting: bool = False
    ) -> None:
        """
        Generate AI response and stream TTS in real-time WITH conversation history.
        Uses PARALLEL TTS PIPELINE (Vapi-style) for ultra-low latency.
        """
        from datetime import datetime, timezone

        try:
            # 👋 HANDLE AUTO-GREETING - Skip LLM, use pre-defined greeting
            if is_greeting:
                # Get greeting from agent or use default
                if self._h.agent and hasattr(self._h.agent, "first_message") and self._h.agent.first_message:
                    greeting_text = self._h.agent.first_message
                else:
                    greeting_text = "hello how are you"

                # Add greeting to transcript
                await self._h._add_to_transcript("agent", greeting_text, "greeting")

                # Queue greeting TTS directly (skip LLM!)
                if not self._h._tts_pipeline:
                    return
                await self._h._tts_pipeline.queue_tts(
                    {
                        "text": greeting_text,
                        "chunk_id": "greeting",
                        "use_ssml": self._h._use_ssml,
                        "is_final": True,
                    }
                )

                # Mark as not primed for the greeting
                self._h._twilio_buffer_primed = False
                return  # Done! No LLM needed for greeting

            # Reset TTS state for new response generation
            self._h._tts_cancel.clear()
            self._h._prev_tts_tail = b""  # Reset crossfade state so new response starts clean
            self._h._elevenlabs_prev_tts_text = ""  # Reset provider continuity state per response turn
            self._h._twilio_buffer_primed = False  # Ensure micro-fade and buffer priming for new utterance

            # Send quick acknowledgement for longer queries (instant from cache!)
            await self.send_quick_acknowledgement(user_text)

            # Build conversation context from transcript
            conversation_history: List[Dict[str, Any]] = []
            if self._h.call_session and self._h.call_session.call_transcript:
                try:
                    raw = self._h.call_session.call_transcript
                    conversation_history = json.loads(raw) if isinstance(raw, str) else list(raw)
                except Exception:
                    conversation_history = []

            # Build history text - bounded filtered history for stable long-call memory
            history_text = ""
            if conversation_history:
                try:
                    history_lines: List[str] = []
                    filtered: List[Tuple[str, str]] = []
                    for msg in conversation_history:
                        if isinstance(msg, Dict):
                            # Handle both 'content' and 'message' keys
                            role = msg.get("role", "unknown")
                            content = msg.get("content") or msg.get("message", "")
                            message_type = msg.get("message_type", "")

                            # Filter: Only include client and agent messages (skip system/greeting/status messages)
                            if (
                                content
                                and role in ["client", "agent"]
                                and message_type not in ["greeting", "system", "status"]
                            ):
                                filtered.append((role, content))

                    # Use only the most recent HISTORY_MAX_MESSAGES to keep prompt within model limits
                    max_msgs = getattr(self._h, "HISTORY_MAX_MESSAGES", VOICE_TUNABLES.history_max_messages)
                    if len(filtered) > max_msgs:
                        filtered = filtered[-max_msgs:]

                    # Build history text from the bounded window
                    for role, content in filtered:
                        history_lines.append(f"{role.capitalize()}: {content}")

                    history_text = "\n".join(history_lines)
                except Exception:
                    history_text = ""

            # Build authoritative business-knowledge block (tenant/agent scoped).
            # Best-effort + timeout-capped so voice latency remains predictable.
            tenant_uuid = self._h.call_session.tenant_id if self._h.call_session else None
            agent_uuid = self._h.agent.id if self._h.agent else None
            business_knowledge_block = ""
            if tenant_uuid:
                try:
                    loop = asyncio.get_running_loop()

                    def _build_business_knowledge_block() -> str:
                        return agent_service.build_business_knowledge_context_block(
                            db=self._h.db,
                            tenant_id=tenant_uuid,
                            agent_id=agent_uuid,
                        )

                    business_knowledge_block = await asyncio.wait_for(
                        loop.run_in_executor(None, _build_business_knowledge_block),
                        timeout=float(
                            getattr(settings, "VOICE_BUSINESS_KB_FETCH_TIMEOUT_SEC", 0.25) or 0.25
                        ),
                    )
                except Exception as exc:
                    logger.debug("Business knowledge fetch skipped: %s", exc)

            # When no business facts loaded, inject an explicit "do not invent" guard.
            _bk_block = business_knowledge_block or (
                "# AUTHORITATIVE BUSINESS FACTS\n"
                "No verified business facts are loaded for this call.\n"
                "CRITICAL: Do NOT invent or assume ANY business details (name, address, phone, "
                "email, services, prices, hours, or any other specifics).\n"
                "If the caller asks about the business, say that specific information is not "
                "available to you right now and offer to help in another way."
            )

            # Build system prompt with agent personality + history
            agent_name = self._h.agent.name if self._h.agent and self._h.agent.name else "AI Assistant"
            agent_language = self._h.agent.language if self._h.agent and self._h.agent.language else "en"
            from app.core.agent_runtime import resolve_tts_runtime

            tts_provider_slug = (
                resolve_tts_runtime(
                    self._h.agent, db=getattr(self._h, "db", None)
                ).adapter_slug
                if self._h.agent
                else ""
            )
            elevenlabs_audio_tags_enabled = supports_elevenlabs_audio_tags(tts_provider_slug)
            if elevenlabs_audio_tags_enabled:
                output_plain_text_rule, no_ssml_rule_base, no_ssml_rule = (
                    get_elevenlabs_voice_prompt_rule_lines()
                )
            else:
                output_plain_text_rule = (
                    "- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML, XML, or any tags. "
                    "Prosody is handled by the system."
                )
                no_ssml_rule_base = (
                    "4. NO SSML: Do NOT output <speak>, <prosody>, or any XML tags. Plain text only."
                )
                no_ssml_rule = "3. NO SSML: Plain text only. No <speak>, <prosody>, or XML."
            elevenlabs_audio_tag_block = build_elevenlabs_audio_tag_prompt_block(tts_provider_slug)

            # Base prompt for phone conversations (voice-first, plain text only, no SSML)
            base_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call with a human.

# STYLE & TONE
- VOICE-FIRST: Your output is for Text-to-Speech. Use short, punchy sentences.
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
- CONCISE: Max 20 words per response unless explaining something complex.
- NO ROBOT TALK: Avoid "As an AI" or formal greetings. Use "Hey," "Hi," or "Hello."
{output_plain_text_rule}
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

# CRITICAL RULES
1. NO REPETITION: If the history shows you asked a question, move to the next point.
2. HANDLING SILENCE: If the user says something vague, ask a clarifying question.
3. TERMINATION: When the objective is met, say a friendly goodbye and end your response with exactly [END_CALL].
4. BUSINESS FACTS: For questions about business name, address, phone, email, website, services, or pricing, use AUTHORITATIVE BUSINESS FACTS below. If details are not present there, say they are not available — do NOT invent them.
5. SERVICE SCOPE: Strictly follow "BUSINESS SCOPE & POLICY — STRICT RULES" in AUTHORITATIVE BUSINESS FACTS. Only offer the services listed there. If asked for anything else, decline politely and offer what we actually do.
6. SERVICE AREA: If Service Areas are listed and restricted, and the caller is outside them, apologize, name the covered areas, say a short goodbye, and end your response with exactly [END_CALL]. If Service Areas describe global/remote/worldwide coverage, never refuse based on location.
{no_ssml_rule_base}

{elevenlabs_audio_tag_block}

{_bk_block}

# GOAL
Continue the conversation based on the history above. Be {agent_name}."""

            # Batch calls may inject a per-row substituted prompt via call_metadata
            batch_prompt_override = None
            ab_prompt_override = None
            if self._h.call_session and self._h.call_session.call_metadata:
                batch_prompt_override = self._h.call_session.call_metadata.get(
                    "batch_prompt_override"
                )
                # A/B prompt testing: variant + resolved prompt text are locked onto
                # call_metadata at dispatch time (see ab_testing_service) and never
                # re-resolved mid-call.
                ab_prompt_override = self._h.call_session.call_metadata.get(
                    "ab_prompt_text"
                )

            # Use agent's custom system prompt if available, otherwise use base prompt
            if self._h.agent and self._h.agent.system_prompt:
                effective_custom_prompt = (
                    batch_prompt_override or ab_prompt_override or self._h.agent.system_prompt
                )
                system_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# GROUNDING RULES (NON-NEGOTIABLE — APPLY BEFORE READING CUSTOM INSTRUCTIONS)
These rules override any conflicting custom instructions below. Never deviate from them.
1. BUSINESS FACTS: Answer questions about business name, address, phone, email, website, services, or pricing ONLY using AUTHORITATIVE BUSINESS FACTS below. Never invent or assume any detail not explicitly written there. If a fact is absent, say it is not available.
2. SERVICE SCOPE: Only offer, quote, or schedule services listed in AUTHORITATIVE BUSINESS FACTS. Politely decline anything outside that list.
3. SERVICE AREA: If Service Areas are listed and restricted, and the caller is outside them, apologize, name the covered areas, and end with [END_CALL]. Never refuse based on location when coverage is global/remote.
4. NO INVENTION: When you are uncertain, say so. Do not fill gaps with guesses.

{_bk_block}

# CUSTOM INSTRUCTIONS
{effective_custom_prompt}

# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
{output_plain_text_rule}
- NO BRACKET TAGS: Never output bracketed tags like [pause], [laugh], [breathes], [excited], [1], [2], or any similar annotation. These will not be rendered — they will be read aloud literally.
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

# CRITICAL RULES
1. NO REPETITION: Do not repeat questions already asked. Move to the next point.
2. TERMINATION: When all objectives from your custom instructions are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{no_ssml_rule}

{elevenlabs_audio_tag_block}

# GOAL
Follow your custom instructions. Continue from the history above. Be {agent_name}."""
            elif self._h.agent and self._h.agent.model and self._h.agent.model.system_prompt:
                effective_model_prompt = (
                    batch_prompt_override
                    or ab_prompt_override
                    or self._h.agent.model.system_prompt
                )
                system_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# GROUNDING RULES (NON-NEGOTIABLE — APPLY BEFORE READING MODEL INSTRUCTIONS)
These rules override any conflicting model instructions below. Never deviate from them.
1. BUSINESS FACTS: Answer questions about business name, address, phone, email, website, services, or pricing ONLY using AUTHORITATIVE BUSINESS FACTS below. Never invent or assume any detail not explicitly written there. If a fact is absent, say it is not available.
2. SERVICE SCOPE: Only offer, quote, or schedule services listed in AUTHORITATIVE BUSINESS FACTS. Politely decline anything outside that list.
3. SERVICE AREA: If Service Areas are listed and restricted, and the caller is outside them, apologize, name the covered areas, and end with [END_CALL]. Never refuse based on location when coverage is global/remote.
4. NO INVENTION: When you are uncertain, say so. Do not fill gaps with guesses.

{_bk_block}

# MODEL INSTRUCTIONS
{effective_model_prompt}

# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use fillers like "uhm," "well," "I see" occasionally.
{output_plain_text_rule}
- NO BRACKET TAGS: Never output bracketed tags like [pause], [laugh], [breathes], [excited], [1], [2], or any similar annotation. These will not be rendered — they will be read aloud literally.

# CONVERSATION STATE
Previous conversation:
{history_text}

# CRITICAL RULES
1. NO REPETITION: Do not repeat questions. Move to the next point.
2. TERMINATION: When all objectives are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{no_ssml_rule}

{elevenlabs_audio_tag_block}

# GOAL
Follow the model instructions. Continue from the history above. Be {agent_name}."""
            else:
                system_prompt = base_prompt

            call_policy_block = agent_service.build_call_policy_block(
                business_knowledge_block=business_knowledge_block,
                transfer_route=getattr(self._h.agent, "transfer_route", None) if self._h.agent else None,
            )
            if call_policy_block:
                system_prompt = call_policy_block + "\n" + system_prompt

            # KB context injection: runs when flow.knowledge_base_ids is non-empty.
            # Injected AFTER the system prompt and BEFORE conversation history.
            kb_context_block = ""
            flow = getattr(self._h, "call_flow", None)
            flow_kb_ids = (flow.knowledge_base_ids or []) if flow else []
            if flow_kb_ids and self._h.db:
                try:
                    from app.services.kb_retrieval_service import retrieve_kb_context_for_turn
                    from app.utils.redis_client import get_redis

                    kb_context_block, kb_latency_ms = await asyncio.wait_for(
                        retrieve_kb_context_for_turn(
                            transcript=user_text,
                            kb_ids=flow_kb_ids,
                            redis_client=get_redis(),
                        ),
                        timeout=0.45,  # stay within 500ms budget; fail open if exceeded
                    )
                    logger.info(
                        "kb_retrieval latency_ms=%.1f kb_count=%d call_sid=%s",
                        kb_latency_ms,
                        len(flow_kb_ids),
                        getattr(self._h, "call_sid", ""),
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "kb_retrieval timed out after 450ms; proceeding without KB context"
                    )
                except Exception as exc:
                    logger.error("kb_retrieval failed; proceeding without context: %s", exc)

            # Inject KB context block between system prompt and conversation history.
            if kb_context_block:
                anchor = "# CONVERSATION STATE"
                if anchor in system_prompt:
                    system_prompt = system_prompt.replace(
                        anchor, kb_context_block + "\n\n" + anchor, 1
                    )
                else:
                    system_prompt = system_prompt + "\n\n" + kb_context_block

            # HubSpot CRM context injection: fetched once at call start (Redis-cached
            # contact lookup) and cached on call_session.call_metadata, so every later
            # turn (the prompt is rebuilt each turn) is a cheap in-memory dict read.
            # Fails open on timeout/error — never blocks the call.
            crm_context_block = ""
            if self._h.call_session and self._h.db:
                try:
                    from app.services.hubspot_service import get_crm_context_block_for_call

                    crm_context_block = await asyncio.wait_for(
                        get_crm_context_block_for_call(self._h.db, self._h.call_session),
                        timeout=0.6,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "HubSpot CRM context lookup timed out; proceeding without CRM context"
                    )
                except Exception as exc:
                    logger.error(
                        "HubSpot CRM context lookup failed; proceeding without context: %s", exc
                    )

            if crm_context_block:
                anchor = "# CONVERSATION STATE"
                if anchor in system_prompt:
                    system_prompt = system_prompt.replace(
                        anchor, crm_context_block + "\n\n" + anchor, 1
                    )
                else:
                    system_prompt = system_prompt + "\n\n" + crm_context_block

            # Cross-session caller memory: fetched once at call start (DB lookup,
            # 100ms timeout budget, fail-open) and cached on call_session.call_metadata
            # so every later turn is a cheap in-memory dict read. Injected right after
            # the KB/CRM context blocks and before conversation history.
            caller_memory_block = ""
            if self._h.call_session and self._h.db and flow and flow.caller_memory_enabled:
                try:
                    from app.services.caller_memory_service import (
                        get_caller_memory_context_block_for_call,
                    )

                    caller_memory_block = await get_caller_memory_context_block_for_call(
                        self._h.db, self._h.call_session, flow
                    )
                except Exception as exc:
                    logger.error(
                        "caller_memory lookup failed; proceeding without context: %s", exc
                    )

            if caller_memory_block:
                anchor = "# CONVERSATION STATE"
                if anchor in system_prompt:
                    system_prompt = system_prompt.replace(
                        anchor, caller_memory_block + "\n\n" + anchor, 1
                    )
                else:
                    system_prompt = system_prompt + "\n\n" + caller_memory_block

            # HubSpot field-mapping substitution: replaces `{prompt_variable}` tokens
            # in the prompt with tenant-configured HubSpot contact field values.
            # Resolved once per call (Redis/DB-cached) and fails open on timeout/error.
            if self._h.call_session and self._h.db:
                try:
                    from app.services.hubspot_service import (
                        apply_field_mapping_values,
                        get_field_mapping_values_for_call,
                    )

                    field_mapping_values = await asyncio.wait_for(
                        get_field_mapping_values_for_call(self._h.db, self._h.call_session),
                        timeout=0.6,
                    )
                    if field_mapping_values:
                        system_prompt = apply_field_mapping_values(
                            system_prompt, field_mapping_values
                        )
                except asyncio.TimeoutError:
                    logger.warning(
                        "HubSpot field mapping lookup timed out; proceeding without substitution"
                    )
                except Exception as exc:
                    logger.error(
                        "HubSpot field mapping lookup failed; proceeding without substitution: %s",
                        exc,
                    )

            from app.core.agent_runtime import llm_service_for_provider, resolve_llm_runtime

            llm_runtime = resolve_llm_runtime(self._h.agent)
            model_name = llm_runtime.model_name
            api_key = llm_runtime.api_key
            temperature = llm_runtime.temperature
            max_tokens = llm_runtime.max_tokens
            llm_service = llm_service_for_provider(llm_runtime.provider_slug)

            # Stream LLM output and QUEUE for PARALLEL TTS PIPELINE (Vapi-style)
            chunk_counter = 0
            _tts_time_flush_s = max(
                0.10,
                float(getattr(settings, "VOICE_TTS_TIME_FLUSH_SEC", 0.15) or 0.15),
            )
            logger.info(
                f"🧠 Calling LLM ({llm_service.__class__.__name__ if hasattr(llm_service, '__class__') else 'Service'}) "
                f"for response to: '{user_text[:20]}...'"
            )

            async def try_stream(service, model: str, api_key_override: Optional[str] = None) -> str:
                nonlocal chunk_counter

                response_accum = ""
                tts_buffer = ""
                end_call_after = False
                transfer_after = False
                _transfer_re = re.compile(r"\[\s*TRANSFER_CALL\s*\]", re.IGNORECASE)
                first_tts_chunk = True
                last_flush_ts = time.perf_counter()

                def _strip_control_tokens(text: str) -> str:
                    if not text:
                        return ""
                    out = text.replace("[END_CALL]", "").replace("[SCREENING_QUALIFIED]", "")
                    out = re.sub(r"\[\s*TRANSFER_CALL\s*\]", "", out, flags=re.IGNORECASE)
                    out = re.sub(r"\[OUTCOME:[^\]]+\]", "", out)
                    out = re.sub(r"\[CHECK_SLOTS:[^\]]*\]", "", out)
                    out = re.sub(r"\[BOOK_APPOINTMENT:[^\]]*\]", "", out)
                    # Strip all known audio tags so they are never spoken as literal words.
                    out = strip_eleven_v3_style_tags_for_non_eleven_tts(out)
                    return out

                def _find_flush_index(buf: str):
                    if not buf:
                        return None

                    nl = buf.find("\n")
                    if nl != -1:
                        prefix = buf[:nl].strip()
                        if len(prefix.split()) >= self._h.TTS_FLUSH_MIN_WORDS:
                            return nl

                    last_boundary = None
                    for m in re.finditer(r"([.!?])(\s+|$)", buf):
                        last_boundary = m.end(1)

                    if last_boundary is not None:
                        prefix = buf[:last_boundary].strip()
                        if len(prefix.split()) >= self._h.TTS_FLUSH_MIN_WORDS:
                            return last_boundary

                    words = buf.split()
                    if len(words) >= self._h.TTS_FLUSH_MAX_WORDS:
                        last_soft = None
                        for m in re.finditer(r"([,;:])(\s+|$)", buf):
                            last_soft = m.end(1)
                        if last_soft is not None:
                            prefix = buf[:last_soft].strip()
                            if len(prefix.split()) >= self._h.TTS_FLUSH_MIN_WORDS:
                                return last_soft
                    return None

                def _find_time_flush_index(buf: str):
                    if not buf:
                        return None
                    words = buf.split()
                    if len(words) < max(self._h.TTS_FLUSH_MIN_WORDS, 5):
                        return None

                    target_words = min(8, len(words))
                    m = re.match(rf"^(?:\S+\s+){{{target_words - 1}}}\S+", buf)
                    if not m:
                        return None
                    return m.end()

                async for chunk in service.stream_text(
                    prompt=user_text,
                    system_prompt=system_prompt,
                    model_name=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key_override,
                ):
                    if not chunk:
                        continue
                    if self._h._tts_cancel.is_set():
                        break

                    response_accum += chunk
                    tts_buffer += chunk

                    if chunk:
                        _vm = getattr(self._h, "_voice_metrics", None)
                        if _vm:
                            _vm.mark_llm_first_token()

                    if "[END_CALL]" in response_accum:
                        end_call_after = True
                        tts_buffer = _strip_control_tokens(tts_buffer)

                    if _transfer_re.search(response_accum):
                        transfer_after = True
                        end_call_after = False
                        tts_buffer = _strip_control_tokens(tts_buffer)

                    if "[OUTCOME:" in tts_buffer:
                        tts_buffer = _strip_control_tokens(tts_buffer)

                    if "[SCREENING_QUALIFIED]" in response_accum:
                        tts_buffer = _strip_control_tokens(tts_buffer)

                    flush_idx = _find_flush_index(tts_buffer)
                    now_ts = time.perf_counter()
                    if flush_idx is None and (now_ts - last_flush_ts) >= _tts_time_flush_s:
                        flush_idx = _find_time_flush_index(tts_buffer)

                    if flush_idx is not None and not self._h._tts_cancel.is_set() and self._h._tts_pipeline:
                        to_speak = tts_buffer[:flush_idx].strip()
                        tts_buffer = tts_buffer[flush_idx:].lstrip()
                        if to_speak:
                            chunk_counter += 1
                            await self._h._tts_pipeline.queue_tts(
                                {
                                    "text": to_speak,
                                    "chunk_id": chunk_counter,
                                    "use_ssml": self._h._use_ssml,
                                    "is_final": False,
                                    "end_call_after": False,
                                }
                            )
                            _vm = getattr(self._h, "_voice_metrics", None)
                            if _vm:
                                _vm.mark_first_tts_queued()
                            last_flush_ts = now_ts
                            first_tts_chunk = False

                # Flush any remaining buffer as final
                full_accum = response_accum.strip()
                end_call_after = end_call_after or ("[END_CALL]" in full_accum)
                if _transfer_re.search(full_accum):
                    transfer_after = True
                    end_call_after = False
                final_text = _strip_control_tokens(tts_buffer).strip()
                if final_text and not self._h._tts_cancel.is_set() and self._h._tts_pipeline:
                    chunk_counter += 1
                    await self._h._tts_pipeline.queue_tts(
                        {
                            "text": final_text,
                            "chunk_id": chunk_counter,
                            "use_ssml": self._h._use_ssml,
                            "is_final": True,
                            "end_call_after": end_call_after and not transfer_after,
                            "transfer_after": transfer_after,
                        }
                    )
                    _vm = getattr(self._h, "_voice_metrics", None)
                    if _vm:
                        _vm.mark_first_tts_queued()
                elif transfer_after and not self._h._tts_cancel.is_set() and self._h._tts_pipeline:
                    chunk_counter += 1
                    await self._h._tts_pipeline.queue_tts(
                        {
                            "text": "One moment.",
                            "chunk_id": chunk_counter,
                            "use_ssml": self._h._use_ssml,
                            "is_final": True,
                            "end_call_after": False,
                            "transfer_after": True,
                        }
                    )
                    _vm = getattr(self._h, "_voice_metrics", None)
                    if _vm:
                        _vm.mark_first_tts_queued()
                return response_accum

            final_text = ""
            try:
                final_text = await try_stream(llm_service, model_name, api_key_override=api_key)
            except Exception as e:
                logger.error(f"LLM streaming failed: {e}", exc_info=True)

            if final_text:
                transcript_text = re.sub(
                    r"\[\s*TRANSFER_CALL\s*\]", "", final_text, flags=re.IGNORECASE
                ).replace("[END_CALL]", "").strip()
                if transcript_text:
                    await self._h._add_to_transcript("agent", transcript_text, "agent_response")

        except Exception as e:
            logger.error(f"Error in generate_and_stream_response: {e}", exc_info=True)

    # ---- High-level entrypoint ----------------------------------------------

    async def on_user_speech(
        self,
        text: str,
        is_final: bool,
        audio_stats: Optional[Dict[str, Any]] = None,
        timestamps: Optional[Dict[str, Any]] = None,
    ) -> ConversationActions:
        """
        High-level decision point for a user speech event.

        Returns a ConversationActions description while also performing
        the underlying side effects (quick-acks, LLM/TTS)
        so the existing handler flow keeps working unchanged.
        """
        actions = ConversationActions()

        if not text:
            return actions

        confidence = float(audio_stats.get("confidence", 0.0)) if audio_stats else 0.0

        if not is_final:
            # Interim path: barge-in, early LLM start.
            await self.process_interim(text, confidence)
            # Reflect whether we decided to start an interim-driven response.
            actions.start_llm_response = bool(getattr(self._h, "_turn_response_started", False))
            return actions

        # Full LLM path matches bidirectional _process_transcript (commit + no duplicate interim)
        await self._h._add_to_transcript("client", text, "speech", confidence)
        self._h._update_booking_memory_from_user_turn(text)
        await self._h._complete_llm_turn_after_stt_final(text, confidence)
        actions.start_llm_response = True
        actions.should_persist_history = True

        return actions

