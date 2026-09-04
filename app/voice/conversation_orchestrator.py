import asyncio
import json
import random
import time
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.core.logger import logger
from app.core.config import settings
from app.services.agent_service import agent_service
from app.services.token_budget_service import token_budget_service
from app.utils.webhook_templating import render_template
from app.utils.eleven_tts_text import (
    strip_eleven_v3_style_tags_for_non_eleven_tts,
)
from app.voice.tts_flush import find_sentence_flush_index, find_time_flush_index
from app.voice.humanization_intent import (
    build_delivery_prompt_block,
    consume_delivery_tag,
    segment_intent_from_tag_attrs,
    strip_delivery_tags,
)
from app.services.llm_circuit_breaker import llm_circuit_breaker

# F-05: Warm transfer phrase pool
_TRANSFER_PHRASES = [
    "Let me connect you with a specialist — just one moment.",
    "I'm going to bring in someone who can help you further — hold just a second.",
    "I'll transfer you now — someone will be right with you.",
]


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

# Shared quick-ack cooldown: how many eligible turns must pass before we
# roll the probability check again after the last ack. Both transports
# reference this one constant (Twilio's BidirectionalStreamHandler exposes
# it as a class attribute sourced from here, for backward-compatible test
# access) so cooldown pacing never silently diverges between them.
QUICK_ACK_COOLDOWN_TURNS = 3

# Shared acknowledgement phrase pool — used identically by both transports.
_QUICK_ACK_PHRASES: Tuple[str, ...] = (
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
)


def decide_quick_ack(
    user_text: str,
    *,
    turns_since_last_ack: int,
    last_phrase: str,
    config: QuickAckConfig | None = None,
    cooldown_turns: int = QUICK_ACK_COOLDOWN_TURNS,
) -> Tuple[str | None, int, str]:
    """
    The ONE shared quick-acknowledgement decision both transports call, so
    cooldown/variety/content gating can never drift between Twilio
    (``BidirectionalStreamHandler._send_quick_acknowledgement``) and
    LiveKit/browser (``ConversationOrchestrator.send_quick_acknowledgement``).

    Applies, in order:
    1. Content: never ack the caller's own short backchannel/confirmation
       ("yes", "okay", "right", ...) — those aren't substantive turns.
    2. Length/skip-phrase eligibility (``should_send_quick_ack``).
    3. Cooldown: only roll the probability check every ``cooldown_turns``
       eligible turns, so acks land occasionally rather than
       (probabilistically) every turn.
    4. Probability (``config.probability``).
    5. Variety: never repeat the exact same phrase back-to-back.

    Pure and side-effect-free: returns
    ``(phrase_or_None, new_turns_since_last_ack, new_last_phrase)`` — it
    never mutates anything. Callers own persisting the two counters onto
    their own per-call state (Twilio: instance attributes; LiveKit:
    ``ConversationOrchestrator`` instance attributes).

    Turn-identity dedup (needed only by Twilio, which can evaluate the same
    logical user turn twice — once on a qualifying interim, once on the STT
    final — see ``bidirectional_stream.py``'s ``_turn_generation_id``) is a
    transport-specific concern and NOT handled here; a caller that
    double-invokes per turn must gate that separately before calling this.
    """
    cfg = config or VOICE_TUNABLES.quick_ack
    text = (user_text or "").strip()
    if not text:
        return None, turns_since_last_ack, last_phrase

    from app.voice.backchannel_classifier import is_known_non_actionable_backchannel

    if is_known_non_actionable_backchannel(text):
        return None, turns_since_last_ack, last_phrase

    if not should_send_quick_ack(text, cfg):
        return None, turns_since_last_ack, last_phrase

    turns_since_last_ack += 1
    if turns_since_last_ack < cooldown_turns:
        return None, turns_since_last_ack, last_phrase

    if random.random() >= cfg.probability:
        return None, turns_since_last_ack, last_phrase

    choices = [p for p in _QUICK_ACK_PHRASES if p != last_phrase] or list(_QUICK_ACK_PHRASES)
    phrase = random.choice(choices)
    return phrase, 0, phrase


# ---------------------------------------------------------------------------
# Shared grounding-rule text (deduplicated — was three near-identical copies).
#
# Names BOTH the AUTHORITATIVE BUSINESS FACTS block (agent_service.
# build_business_knowledge_context_block) and the KNOWLEDGE BASE CONTEXT block
# (kb_retrieval_service.retrieve_kb_context_for_turn) as usable/authoritative
# sources — previously only the former was mentioned, so the model had no
# instruction telling it the KB block (when present) was safe to use, even
# though it was already spliced into the prompt.
# ---------------------------------------------------------------------------
_BUSINESS_FACTS_GROUNDING_TEXT = (
    "Answer questions about business name, address, phone, email, website, services, "
    "or pricing using AUTHORITATIVE BUSINESS FACTS and, when present in this prompt, "
    "KNOWLEDGE BASE CONTEXT — both are authoritative sources for this call, regardless "
    "of where each appears in this prompt. "
    "Never invent or assume any detail that is not explicitly written in one of those "
    "two sources. If a fact is absent from both, say it is not available."
)


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


def rag_prefetch_matches_final(prefetch_source_text: str, final_text: str) -> bool:
    """
    Decide whether a RAG/KB prefetch fired on an STT *interim* is still safe
    to reuse for the STT *final* of the same turn.

    The Twilio path (bidirectional_stream.py's `_prefetch_rag_context`) fires
    on the first qualifying interim and always reuses whatever result that
    produced, with no divergence check — it accepts that a slight interim/
    final wording difference is fine for retrieval purposes. The browser path
    adds this lightweight safety check on top of that same pattern: if the
    final utterance diverges *materially* from the interim that triggered the
    prefetch (e.g. the STT correction changed the meaning, not just a couple
    of trailing words), the prefetch result is discarded and a fresh
    retrieval is used instead — never a behavior change to Twilio, since this
    helper is only consumed by the browser call path.
    """
    source = (prefetch_source_text or "").strip().lower()
    final = (final_text or "").strip().lower()
    if not source or not final:
        return False
    if final.startswith(source) or source.startswith(final):
        return True
    source_words = set(source.split())
    final_words = set(final.split())
    overlap = len(source_words & final_words) / max(len(source_words), len(final_words))
    return overlap >= 0.6


@dataclass
class ConversationActions:
    """
    High-level actions decided from a user speech event.
    The handler uses this to drive quick-acks, LLM responses, and history updates.
    """

    quick_ack_text: str | None = None
    start_llm_response: bool = False
    end_call_after: bool = False

    # Updated conversation history (already windowed)
    updated_history: List[Dict[str, Any]] | None = None
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

        # V-07 History Summarization Pipeline state.
        # _history_summary holds the rolling compressed summary of conversation turns
        # that have been dropped from the active sliding window.  It is prepended to the
        # history block in build_system_prompt so the agent retains full-call memory.
        # _last_summarized_turn_index tracks the exclusive end of the already-summarized
        # range so turns are never compressed twice.
        self._history_summary: str = ""
        self._last_summarized_turn_index: int = 0

        # Quick-ack cooldown/variety state (see decide_quick_ack) — per-call,
        # in-memory only, mirroring Twilio's equivalent instance attributes
        # so both transports enforce identical pacing/repetition behavior.
        self._quick_ack_turns_since_last: int = 0
        self._quick_ack_last_phrase: str = ""

    # ---- Interim processing / barge-in gating -------------------------

    async def process_interim(self, transcript: str, confidence: float) -> None:
        """
        Delegates to the bidirectional stream handler (single source of truth).
        """
        await self._h._maybe_process_interim(transcript, confidence)

    # ---- Quick acknowledgements ---------------------------------------

    async def send_quick_acknowledgement(self, user_text: str) -> None:
        """
        Send instant acknowledgement for longer queries while generating full
        response. Delegates the actual decision (content/skip-phrase
        eligibility, cooldown, probability, phrase variety) to the shared
        ``decide_quick_ack`` — the SAME function Twilio's
        ``BidirectionalStreamHandler._send_quick_acknowledgement`` calls —
        so acknowledgement pacing/repetition never diverges between
        transports. (Twilio additionally gates on turn-identity because it
        can evaluate the same logical turn twice; this path calls
        ``generate_and_stream_response`` only once per STT-final turn, so no
        equivalent gate is needed here.)
        """
        ack, self._quick_ack_turns_since_last, new_last_phrase = decide_quick_ack(
            user_text,
            turns_since_last_ack=self._quick_ack_turns_since_last,
            last_phrase=self._quick_ack_last_phrase,
        )
        if not ack:
            return
        if not self._h._tts_pipeline:
            return
        self._quick_ack_last_phrase = new_last_phrase
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

    async def build_system_prompt(self, user_text: str, confidence: float) -> str:
        """
        Assemble the full system prompt for a turn: base/agent-override/
        model-override branching, business-facts grounding, call-policy
        block, RAG/KB injection (prefetch-consume-once), CRM context, caller
        memory, and HubSpot field-mapping substitution.

        Behavior-preserving extraction from ``generate_and_stream_response``
        (which used to build this prompt inline) — reused as-is by that
        method's streaming-text path AND by the Gemini Live native-audio
        session's ``system_instruction`` at session start (see
        ``VoiceOrchestrator._start_gemini_live_session``). ``confidence`` is
        accepted for signature symmetry with the STT-final callback that
        triggers a turn but is not currently read by any branch below.
        """
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

        # V-07: Prepend rolling summary of dropped turns so the agent retains context
        # from earlier in the call (caller name, problem, location, etc.) that has
        # already scrolled out of the active window.  Applied unconditionally outside the
        # transcript-parse guard so it survives both empty-transcript and parse-error paths.
        if self._history_summary:
            summary_block = (
                f"<earlier_conversation_summary>\n"
                f"{self._history_summary.strip()}\n"
                f"</earlier_conversation_summary>\n\n"
            )
            history_text = summary_block + history_text

        # Build system prompt with agent personality + history
        agent_name = self._h.agent.name if self._h.agent and self._h.agent.name else "AI Assistant"
        agent_language = self._h.agent.language if self._h.agent and self._h.agent.language else "en"

        # Provider-independent: the LLM never emits bracketed tags itself under the
        # V-08 architecture. Only the realization layer (tts_provider_capabilities.py)
        # adds real provider-specific tags after the fact.
        output_plain_text_rule = (
            "- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML, XML, or any tags. "
            "Prosody is handled by the system."
        )
        no_ssml_rule_base = (
            "4. NO SSML: Do NOT output <speak>, <prosody>, or any XML tags. Plain text only."
        )
        no_ssml_rule = "3. NO SSML: Plain text only. No <speak>, <prosody>, or XML."
        no_bracket_tags_line = (
            "- NO BRACKET TAGS: Never output bracketed tags like [pause], [laugh], [breathes], "
            "[excited], [1], [2], or any similar annotation. These will not be rendered — they "
            "will be read aloud literally."
        )

        # Base prompt for phone conversations (voice-first, plain text only, no SSML)
        base_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call with a human.

# STYLE & TONE
- VOICE-FIRST: Your output is for Text-to-Speech. Use short, punchy sentences.
- NATURAL: Speak naturally and conversationally. Answer directly. Do not add artificial hesitation, filler words, acknowledgements, or conversational padding unless they are genuinely appropriate to the context.
- STRUCTURED INFO PACING: When reading a phone number, confirmation code, or similar digit sequence aloud, group the digits the way a person naturally would when saying them out loud, separated by commas (e.g. "five five five, one two three, four five six seven"), never as one fast unbroken string of digits.
- TRANSITION PACING: Right after giving the caller a phone number, email address, or other specific detail they asked for, add one short acknowledgement word ("Great," "Got it," "Perfect,") before your next question — this is exactly the kind of genuinely-appropriate acknowledgement the NATURAL rule above allows. Do not move straight from the detail into the next sentence with no beat in between.
- CONCISE: Max 20 words per response unless explaining something complex.
- NO ROBOT TALK: Avoid "As an AI" or formal greetings. Use "Hey," "Hi," or "Hello."
{output_plain_text_rule}
{no_bracket_tags_line}
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

# CRITICAL RULES
1. NO REPETITION: If the history shows you asked a question, move to the next point. Any information the caller already gave is still valid — do not ask for it again or re-confirm it once acknowledged.
2. HANDLING SILENCE: If the user says something vague, ask a clarifying question.
3. TERMINATION: When the objective is met, say a friendly goodbye and end your response with exactly [END_CALL].
4. BUSINESS FACTS: {_BUSINESS_FACTS_GROUNDING_TEXT}
5. SERVICE SCOPE: Strictly follow "BUSINESS SCOPE & POLICY — STRICT RULES" in AUTHORITATIVE BUSINESS FACTS. Only offer the services listed there. If asked for anything else, decline politely and offer what we actually do.
6. SERVICE AREA: If Service Areas are listed and restricted, and the caller is outside them, apologize, name the covered areas, say a short goodbye, and end your response with exactly [END_CALL]. If Service Areas describe global/remote/worldwide coverage, never refuse based on location.
{no_ssml_rule_base}

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

        # Resolve call flow prompt override if present on session
        flow_prompt_override = None
        call_flow = getattr(self._h, "call_flow", None)
        if call_flow:
            if getattr(call_flow, "current_prompt", None) and call_flow.current_prompt.prompt_text:
                flow_prompt_override = call_flow.current_prompt.prompt_text
            elif call_flow.current_prompt_id and getattr(self._h, "db", None):
                try:
                    from sqlalchemy import select
                    from app.models.prompt_version import PromptVersion
                    pv = self._h.db.execute(
                        select(PromptVersion).where(PromptVersion.id == call_flow.current_prompt_id)
                    ).scalar_one_or_none()
                    if pv and pv.prompt_text:
                        flow_prompt_override = pv.prompt_text
                except Exception as exc:
                    logger.debug("Could not resolve call flow current_prompt: %s", exc)

        # Use agent's custom system prompt / flow prompt if available, otherwise use base prompt
        if (self._h.agent and self._h.agent.system_prompt) or flow_prompt_override:
            effective_custom_prompt = (
                batch_prompt_override
                or ab_prompt_override
                or flow_prompt_override
                or (self._h.agent.system_prompt if self._h.agent else None)
            )
            system_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# GROUNDING RULES (NON-NEGOTIABLE — APPLY BEFORE READING CUSTOM INSTRUCTIONS)
These rules override any conflicting custom instructions below. Never deviate from them.
1. BUSINESS FACTS: {_BUSINESS_FACTS_GROUNDING_TEXT}
2. SERVICE SCOPE: Only offer, quote, or schedule services listed in AUTHORITATIVE BUSINESS FACTS. Politely decline anything outside that list.
3. SERVICE AREA: If Service Areas are listed and restricted, and the caller is outside them, apologize, name the covered areas, and end with [END_CALL]. Never refuse based on location when coverage is global/remote.
4. NO INVENTION: When you are uncertain, say so. Do not fill gaps with guesses.
5. NO CONFIRMATION LOOPS: Once you have acknowledged a piece of information from the caller (e.g. said "Got it"), treat it as captured for the rest of the call. Never ask for or re-confirm that same field again unless the caller explicitly says it was wrong. If you notice you are about to ask about a field you already acknowledged, do not — move to the next unconfirmed item, or if all required items are acknowledged, summarize and proceed to close the call instead.

# CUSTOM INSTRUCTIONS
{effective_custom_prompt}

# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Speak naturally and conversationally. Answer directly. Do not add artificial hesitation, filler words, acknowledgements, or conversational padding unless they are genuinely appropriate to the context.
- STRUCTURED INFO PACING: When reading a phone number, confirmation code, or similar digit sequence aloud, group the digits the way a person naturally would when saying them out loud, separated by commas (e.g. "five five five, one two three, four five six seven"), never as one fast unbroken string of digits.
- TRANSITION PACING: Right after giving the caller a phone number, email address, or other specific detail they asked for, add one short acknowledgement word ("Great," "Got it," "Perfect,") before your next question — this is exactly the kind of genuinely-appropriate acknowledgement the NATURAL rule above allows. Do not move straight from the detail into the next sentence with no beat in between.
{output_plain_text_rule}
{no_bracket_tags_line}
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

# CRITICAL RULES
1. CONVERSATION CONTINUITY: Read "Previous conversation" above before every reply. Any information the caller already gave is still valid — do not ask for it again or re-confirm it once you have acknowledged it (e.g. said "Got it"). Do not restart your intake flow from the beginning mid-call.
2. NO REPETITION: Do not repeat questions already asked. Move to the next point.
3. TERMINATION: When all objectives from your custom instructions are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{no_ssml_rule}

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
1. BUSINESS FACTS: {_BUSINESS_FACTS_GROUNDING_TEXT}
2. SERVICE SCOPE: Only offer, quote, or schedule services listed in AUTHORITATIVE BUSINESS FACTS. Politely decline anything outside that list.
3. SERVICE AREA: If Service Areas are listed and restricted, and the caller is outside them, apologize, name the covered areas, and end with [END_CALL]. Never refuse based on location when coverage is global/remote.
4. NO INVENTION: When you are uncertain, say so. Do not fill gaps with guesses.
5. NO CONFIRMATION LOOPS: Once you have acknowledged a piece of information from the caller (e.g. said "Got it"), treat it as captured for the rest of the call. Never ask for or re-confirm that same field again unless the caller explicitly says it was wrong. If you notice you are about to ask about a field you already acknowledged, do not — move to the next unconfirmed item, or if all required items are acknowledged, summarize and proceed to close the call instead.

# MODEL INSTRUCTIONS
{effective_model_prompt}

# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Speak naturally and conversationally. Answer directly. Do not add artificial hesitation, filler words, acknowledgements, or conversational padding unless they are genuinely appropriate to the context.
- STRUCTURED INFO PACING: When reading a phone number, confirmation code, or similar digit sequence aloud, group the digits the way a person naturally would when saying them out loud, separated by commas (e.g. "five five five, one two three, four five six seven"), never as one fast unbroken string of digits.
- TRANSITION PACING: Right after giving the caller a phone number, email address, or other specific detail they asked for, add one short acknowledgement word ("Great," "Got it," "Perfect,") before your next question — this is exactly the kind of genuinely-appropriate acknowledgement the NATURAL rule above allows. Do not move straight from the detail into the next sentence with no beat in between.
{output_plain_text_rule}
{no_bracket_tags_line}

# CONVERSATION STATE
Previous conversation:
{history_text}

# CRITICAL RULES
1. CONVERSATION CONTINUITY: Read "Previous conversation" above before every reply. Any information the caller already gave is still valid — do not ask for it again or re-confirm it once you have acknowledged it (e.g. said "Got it"). Do not restart your intake flow from the beginning mid-call.
2. NO REPETITION: Do not repeat questions. Move to the next point.
3. TERMINATION: When all objectives are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{no_ssml_rule}

# GOAL
Follow the model instructions. Continue from the history above. Be {agent_name}."""
        else:
            system_prompt = base_prompt

        call_policy_block = agent_service.build_call_policy_block(
            transfer_route=getattr(self._h.agent, "transfer_route", None) if self._h.agent else None,
        )
        if call_policy_block:
            system_prompt = call_policy_block + "\n" + system_prompt

        # KB context injection: runs when flow.knowledge_base_ids is non-empty.
        # Injected AFTER the system prompt and BEFORE conversation history.
        #
        # RAG retrieval: reuse the interim-fired prefetch when available
        # (fired in LiveKitBrowserCallHandler._maybe_start_rag_prefetch on
        # STT interim, overlapping STT endpointing) instead of always
        # blocking here on a fresh retrieval — mirrors
        # bidirectional_stream.py's prefetch-consume-once pattern, adapted
        # to this transport's kb_retrieval_service call. Falls back to the
        # exact same synchronous retrieval this code path always used
        # when no prefetch fired (e.g. a very short first utterance) or
        # when the final utterance diverged materially from the interim
        # that triggered the prefetch.
        kb_context_block = ""
        flow = getattr(self._h, "call_flow", None)
        flow_kb_ids = (flow.knowledge_base_ids or []) if flow else []
        if flow_kb_ids and self._h.db:
            _kb_timeout_sec = float(
                getattr(settings, "RAG_KB_RETRIEVAL_TIMEOUT_SEC", 0.7) or 0.7
            )

            # Consume the prefetch slot exactly once per turn, immediately —
            # so it can never be read/reused by a later, unrelated turn.
            # isinstance-checked (not just `is not None`) so a handler
            # that never initialised this attribute at all (e.g. a bare
            # test double) is treated the same as "no prefetch fired",
            # rather than accidentally treating an unrelated truthy value
            # as a real in-flight prefetch task.
            _prefetch = getattr(self._h, "_rag_prefetch_task", None)
            if not isinstance(_prefetch, asyncio.Task):
                _prefetch = None
            self._h._rag_prefetch_task = None
            _prefetch_source_text = getattr(self._h, "_rag_prefetch_source_text", "")
            if not isinstance(_prefetch_source_text, str):
                _prefetch_source_text = ""
            self._h._rag_prefetch_source_text = ""

            if _prefetch is not None and not rag_prefetch_matches_final(
                _prefetch_source_text, user_text
            ):
                # Final utterance materially diverged from the interim that
                # triggered this prefetch — don't reuse a possibly-wrong
                # result. Cancel it (if still running) and fall through to
                # a fresh retrieval below, same as the "no prefetch" branch.
                if not _prefetch.done():
                    _prefetch.cancel()
                _prefetch = None

            try:
                if _prefetch is not None:
                    if _prefetch.done():
                        # Already finished — zero-cost read. Check for a stored
                        # exception explicitly rather than relying on
                        # _prefetch_kb_context's current "always returns
                        # ('', 0.0), never raises" contract — that contract is
                        # easy to break in a future edit, and an unguarded
                        # .result() would then raise here and be silently
                        # swallowed by the outer except Exception below.
                        _prefetch_exc = _prefetch.exception()
                        if _prefetch_exc is not None:
                            logger.debug(
                                "[RAG prefetch] stored exception: %s", _prefetch_exc
                            )
                            kb_context_block, kb_latency_ms = "", 0.0
                        else:
                            kb_context_block, kb_latency_ms = _prefetch.result()
                            logger.debug("[RAG prefetch] used prefetch result (done)")
                    else:
                        # Still running — await the SAME in-flight task
                        # (never start a second parallel retrieval).
                        # asyncio.shield so a local timeout here doesn't
                        # cancel a retrieval that may still be useful to
                        # warm the KB cache for a future turn.
                        kb_context_block, kb_latency_ms = await asyncio.wait_for(
                            asyncio.shield(_prefetch),
                            timeout=_kb_timeout_sec,
                        )
                        logger.debug("[RAG prefetch] awaited in-flight prefetch")
                else:
                    from app.services.kb_retrieval_service import retrieve_kb_context_for_turn
                    from app.utils.redis_client import get_redis

                    kb_context_block, kb_latency_ms = await asyncio.wait_for(
                        retrieve_kb_context_for_turn(
                            transcript=user_text,
                            kb_ids=flow_kb_ids,
                            redis_client=get_redis(),
                        ),
                        timeout=_kb_timeout_sec,  # fail open if exceeded — see settings.RAG_KB_RETRIEVAL_TIMEOUT_SEC
                    )
                logger.info(
                    "kb_retrieval latency_ms=%.1f kb_count=%d call_sid=%s",
                    kb_latency_ms,
                    len(flow_kb_ids),
                    getattr(self._h, "call_sid", ""),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "kb_retrieval timed out after %.0fms; proceeding without KB context",
                    _kb_timeout_sec * 1000,
                )
            except asyncio.CancelledError:
                # This turn itself was cancelled (e.g. barge-in) while
                # awaiting the prefetch — propagate rather than swallow.
                # The shielded prefetch task (if any) keeps running
                # independently and is not affected by this.
                raise
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

        # Salesforce CRM context injection: same fail-open, once-per-call,
        # call_metadata-cached pattern as the HubSpot block above.
        salesforce_context_block = ""
        if self._h.call_session and self._h.db:
            try:
                from app.services import salesforce_service

                salesforce_context_block = await asyncio.wait_for(
                    salesforce_service.get_crm_context_block_for_call(
                        self._h.db, self._h.call_session
                    ),
                    timeout=0.6,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Salesforce CRM context lookup timed out; proceeding without CRM context"
                )
            except Exception as exc:
                logger.error(
                    "Salesforce CRM context lookup failed; proceeding without context: %s", exc
                )

        if salesforce_context_block:
            anchor = "# CONVERSATION STATE"
            if anchor in system_prompt:
                system_prompt = system_prompt.replace(
                    anchor, salesforce_context_block + "\n\n" + anchor, 1
                )
            else:
                system_prompt = system_prompt + "\n\n" + salesforce_context_block

        # GoHighLevel (GHL) CRM context injection: same fail-open, once-per-call,
        # call_metadata-cached pattern as the HubSpot/Salesforce blocks above.
        ghl_context_block = ""
        if self._h.call_session and self._h.db:
            try:
                from app.services import ghl_service

                ghl_context_block = await asyncio.wait_for(
                    ghl_service.get_crm_context_block_for_call(
                        self._h.db, self._h.call_session
                    ),
                    timeout=0.6,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "GHL CRM context lookup timed out; proceeding without CRM context"
                )
            except Exception as exc:
                logger.error(
                    "GHL CRM context lookup failed; proceeding without context: %s", exc
                )

        if ghl_context_block:
            anchor = "# CONVERSATION STATE"
            if anchor in system_prompt:
                system_prompt = system_prompt.replace(
                    anchor, ghl_context_block + "\n\n" + anchor, 1
                )
            else:
                system_prompt = system_prompt + "\n\n" + ghl_context_block

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

        # Backend-owned contact intake ("already collected, do not re-ask")
        # block — deterministic complement to the "NO CONFIRMATION LOOPS"
        # grounding rule above. Empty string when nothing is confirmed yet,
        # so calls that never trigger contact intake see no prompt change.
        contact_intake_block = ""
        if self._h.call_session:
            try:
                from app.services.call_session_contact_state import (
                    build_contact_intake_prompt_block,
                    get_contact_intake,
                )

                contact_intake_block = build_contact_intake_prompt_block(
                    get_contact_intake(self._h.call_session)
                )
            except Exception as exc:
                logger.debug("Contact intake prompt block build failed: %s", exc)

        if contact_intake_block:
            anchor = "# CONVERSATION STATE"
            if anchor in system_prompt:
                system_prompt = system_prompt.replace(
                    anchor, contact_intake_block + "\n\n" + anchor, 1
                )
            else:
                system_prompt = system_prompt + "\n\n" + contact_intake_block

        # F-08: Returning caller greeting differentiation
        if caller_memory_block and caller_memory_block.strip():
            returning_caller_instruction = (
                "\n\nIMPORTANT — RETURNING CALLER: This caller has contacted us before. "
                "Open by warmly acknowledging you recognize them — use their name if known "
                "and naturally reference their last interaction (e.g. 'Good to hear from you "
                "again — last time you were asking about X'). Do NOT use a generic opening greeting.\n"
            )
        else:
            returning_caller_instruction = ""
        if returning_caller_instruction:
            system_prompt = system_prompt + returning_caller_instruction

        # F-11: Call-drop reconnect recognition
        _reconnect_call_session = self._h.call_session
        is_reconnect = bool(
            _reconnect_call_session
            and _reconnect_call_session.call_metadata
            and _reconnect_call_session.call_metadata.get("is_reconnect")
        )
        if is_reconnect:
            reconnect_instruction = (
                "\n\nIMPORTANT — RECONNECTING CALLER (CALL DROP): This caller was "
                "disconnected less than 5 minutes ago. Acknowledge the reconnection "
                "warmly (e.g. \"Welcome back! Looks like we got cut off — let's pick "
                "up right where we left off.\"). Do NOT use a generic introductory "
                "greeting, re-introduce yourself, or ask questions that were already "
                "answered in the previous interaction.\n"
            )
            # Best-effort: pull a brief snippet from the dropped session's transcript,
            # if the DB/parent session is reachable — fails open, never blocks the turn.
            if self._h.db:
                try:
                    from app.services.call_session_service import (
                        call_session_service as _css,
                    )

                    reconnect_from_id = _reconnect_call_session.call_metadata.get(
                        "reconnect_from_session_id"
                    )
                    dropped = (
                        _css.get_call_session_by_id_and_tenant(
                            self._h.db,
                            uuid.UUID(reconnect_from_id),
                            _reconnect_call_session.tenant_id,
                        )
                        if reconnect_from_id
                        else None
                    )
                    if dropped and dropped.call_transcript:
                        raw = dropped.call_transcript
                        dropped_history = (
                            json.loads(raw) if isinstance(raw, str) else list(raw)
                        )
                        snippet_lines = []
                        for msg in dropped_history[-4:]:
                            if isinstance(msg, dict):
                                role = msg.get("role", "unknown")
                                content = msg.get("content") or msg.get("message", "")
                                if content:
                                    snippet_lines.append(f"{role.capitalize()}: {content}")
                        if snippet_lines:
                            reconnect_instruction += (
                                "\nContext from the dropped call (for your reference only, "
                                "do not read this verbatim):\n"
                                + "\n".join(snippet_lines)
                                + "\n"
                            )
                except Exception as exc:
                    logger.debug(
                        "F-11 reconnect transcript snippet lookup failed: %s", exc
                    )
            system_prompt = system_prompt + reconnect_instruction

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

        # System Webhooks: inject any {{key}} variables returned by the Pre-Inbound
        # Call Webhook (see app/services/system_webhook_service.py). No-op (cheap,
        # safe) when call_metadata has no webhook_variables — never raises.
        try:
            call_session = getattr(self._h, "call_session", None)
            _webhook_vars = (
                call_session.call_metadata.get("webhook_variables", {})
                if call_session and call_session.call_metadata
                else {}
            )
            if _webhook_vars:
                system_prompt = render_template(system_prompt, _webhook_vars)
        except Exception as exc:
            logger.debug(
                "System webhook variable injection (system prompt) failed: %s", exc
            )

        # V-08: shared `# DELIVERY` instruction block (identical function
        # also called by bidirectional_stream.py's prompt builder) — "" and
        # a complete no-op unless VOICE_ENABLE_LLM_HUMANIZATION is on.
        _delivery_block = build_delivery_prompt_block()
        if _delivery_block:
            system_prompt = system_prompt + "\n\n" + _delivery_block

        return system_prompt

    async def generate_and_stream_response(
        self, user_text: str, confidence: float, is_greeting: bool = False
    ) -> None:
        """
        Generate AI response and stream TTS in real-time WITH conversation history.
        Uses PARALLEL TTS PIPELINE (Vapi-style) for ultra-low latency.
        """

        try:
            # Ambient per-turn context for TtsPipeline's humanization decision
            # (app.voice.humanization_engine) — read via getattr, never required,
            # so any change here can't break TTS. Set on the shared call handler
            # (self._h), the same object TtsPipeline._handler references.
            self._h._current_turn_user_text = user_text
            self._h._current_turn_stt_confidence = confidence

            # 👋 HANDLE AUTO-GREETING - Skip LLM, use pre-defined greeting
            if is_greeting:
                # F-11: Call-drop reconnect recognition — override the scripted
                # greeting entirely when this call was linked as a reconnect by
                # voice_inbound (app/routers/voice.py). Fails open: any lookup
                # issue here just falls through to the normal greeting logic.
                _greeting_call_session = getattr(self._h, "call_session", None)
                is_reconnect = bool(
                    _greeting_call_session
                    and _greeting_call_session.call_metadata
                    and _greeting_call_session.call_metadata.get("is_reconnect")
                )
                if is_reconnect:
                    greeting_text = (
                        "Welcome back! Looks like we got cut off there — let's "
                        "pick up right where we left off."
                    )
                else:
                    # Get greeting from agent or use default. Prefer greeting_message
                    # (current field) over the legacy first_message, matching the
                    # Twilio path's convention (BidirectionalStreamHandler.generate_and_stream_response).
                    greeting_text = None
                    if self._h.agent:
                        greeting_text = (getattr(self._h.agent, "greeting_message", None) or "").strip() or None
                        if not greeting_text:
                            greeting_text = (getattr(self._h.agent, "first_message", None) or "").strip() or None
                    if not greeting_text:
                        greeting_text = "hello how are you"

                # System Webhooks: inject any {{key}} variables returned by the
                # Pre-Inbound Call Webhook before this text reaches any hand-off
                # (native Gemini/OpenAI Realtime session or TtsPipeline below).
                try:
                    call_session = getattr(self._h, "call_session", None)
                    _webhook_vars = (
                        call_session.call_metadata.get("webhook_variables", {})
                        if call_session and call_session.call_metadata
                        else {}
                    )
                    if _webhook_vars:
                        greeting_text = render_template(greeting_text, _webhook_vars)
                except Exception as exc:
                    logger.debug(
                        "System webhook variable injection (greeting) failed: %s", exc
                    )

                # Add greeting to transcript
                await self._h._add_to_transcript("agent", greeting_text, "greeting")

                # Gemini Live native-audio calls must never invoke the external
                # TTS provider — route the greeting through the live session's
                # own send_text so Gemini speaks it in its native voice,
                # instead of queue_tts below. If the session isn't up yet,
                # skip the scripted greeting entirely rather than falling
                # through to queue_tts()/TtsPipeline. Mirrors the same gate in
                # BidirectionalStreamHandler.generate_and_stream_response.
                _vo = getattr(self._h, "_voice_orchestrator", None)
                if _vo is not None and getattr(_vo, "_is_gemini_live", False):
                    session = _vo._gemini_live_session
                    if session is not None:
                        try:
                            await session.send_text(greeting_text, turn_complete=True)
                        except Exception as exc:
                            logger.warning(
                                "[GeminiLive] failed to send greeting via live session: %s", exc
                            )
                    else:
                        logger.debug(
                            "[GeminiLive] skipping scripted greeting — live session not ready yet"
                        )
                    self._h._twilio_buffer_primed = False
                    return

                # Mirror-image bypass for OpenAI Realtime native-audio calls —
                # same rationale as the Gemini Live branch above.
                if _vo is not None and getattr(_vo, "_is_openai_realtime", False):
                    session = _vo._openai_realtime_session
                    if session is not None:
                        try:
                            await session.send_text(greeting_text, respond=True)
                        except Exception as exc:
                            logger.warning(
                                "[OpenAIRealtime] failed to send greeting via live session: %s", exc
                            )
                    else:
                        logger.debug(
                            "[OpenAIRealtime] skipping scripted greeting — live session not ready yet"
                        )
                    self._h._twilio_buffer_primed = False
                    return

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

            if self._h.call_session and self._h.db:
                try:
                    within_budget, _usage, _limit = await token_budget_service.check_daily_budget(
                        self._h.db, self._h.call_session.tenant_id
                    )
                except Exception as exc:
                    logger.warning("Token budget check failed; failing open: %s", exc)
                    within_budget = True
                if not within_budget:
                    refusal_text = (
                        "This workspace has reached its daily AI usage limit. Please "
                        "contact support or your administrator."
                    )
                    await self._h._add_to_transcript("agent", refusal_text, "system")
                    if self._h._tts_pipeline:
                        await self._h._tts_pipeline.queue_tts({
                            "text": refusal_text,
                            "chunk_id": "budget_exceeded",
                            "use_ssml": self._h._use_ssml,
                            "is_final": True,
                        })
                    self._h._twilio_buffer_primed = False
                    return

            # Reset TTS state for new response generation
            self._h._tts_cancel.clear()
            self._h._prev_tts_tail = b""  # Reset crossfade state so new response starts clean
            if hasattr(self._h, "_tts_play_start_ts"):
                self._h._tts_play_start_ts = 0.0  # Clear dead-zone anchor from previous utterance
            # Reset ElevenLabs `previous_text` continuity tracking (Phase 4D-2).
            # Unconditional, every turn — NOT just on barge-in — because
            # cancel_current_and_clear_queue()'s reset only fires on an actual
            # barge-in path (e.g. LiveKitBrowserCallHandler._process_transcript()
            # skips it entirely on a normal, non-barge-in turn). Shared by both
            # transports since this orchestrator is transport-agnostic.
            if self._h._tts_pipeline:
                self._h._tts_pipeline.reset_previous_text_continuity()
            self._h._twilio_buffer_primed = False  # Ensure micro-fade and buffer priming for new utterance

            # Send quick acknowledgement for longer queries (instant from cache!)
            await self.send_quick_acknowledgement(user_text)

            system_prompt = await self.build_system_prompt(user_text, confidence)

            from app.core.agent_runtime import llm_service_for_provider, resolve_llm_runtime

            llm_runtime = resolve_llm_runtime(self._h.agent)
            model_name = llm_runtime.model_name
            api_key = llm_runtime.api_key
            temperature = llm_runtime.temperature
            max_tokens = llm_runtime.max_tokens
            llm_service = llm_service_for_provider(llm_runtime.provider_slug)

            # A-05: distributed circuit breaker — trips OPEN across all calls/workers
            # after consecutive failures, so a doomed provider isn't retried per-turn.
            # Keyed by the actual resolved provider slug (openai/gemini/groq), not a
            # binary gemini/else split — collapsing Groq into "openai" would let an
            # OpenAI outage fast-fail healthy Groq tenants and vice versa.
            _primary_provider = (llm_runtime.provider_slug or "openai").lower()
            _attempted_primary = True
            if not await llm_circuit_breaker.can_execute(_primary_provider):
                logger.warning(
                    "[CircuitBreaker] %s circuit is OPEN — fast-failing to secondary provider without dead air",
                    _primary_provider,
                )
                from app.services.openai_service import openai_service
                from app.services.vertex_gemini_service import vertex_gemini_service

                # Swap to whichever of the two providers this transport fully
                # supports isn't the tripped one. Mirrors bidirectional_stream.py's
                # existing non-gemini fallback swap (only "openai" primaries route
                # to gemini; gemini and groq primaries both route to openai).
                if _primary_provider == "openai":
                    llm_service = vertex_gemini_service
                    model_name = "gemini-2.5-flash"
                else:
                    llm_service = openai_service
                    model_name = "gpt-3.5-turbo"
                api_key = None
                _attempted_primary = False

            # Stream LLM output and QUEUE for PARALLEL TTS PIPELINE (Vapi-style)
            chunk_counter = 0
            _tts_time_flush_s = max(
                0.10,
                float(getattr(settings, "VOICE_TTS_TIME_FLUSH_SEC", 0.15) or 0.15),
            )
            logger.info(
                "🧠 Calling LLM (%s) for response to: '%s...'", llm_service.__class__.__name__ if hasattr(llm_service, '__class__') else 'Service', user_text[:20]
            )
            logger.debug(
                "[LLM] request sent: provider=%s model=%s user_text_len=%s",
                llm_runtime.provider_slug, model_name, len(user_text or ""),
            )

            async def try_stream(service, model: str, api_key_override: str | None = None) -> str:
                nonlocal chunk_counter

                response_accum = ""
                tts_buffer = ""
                end_call_after = False
                transfer_after = False
                _transfer_re = re.compile(r"\[\s*TRANSFER_CALL\s*\]", re.IGNORECASE)
                last_flush_ts = time.perf_counter()
                # V-07: fired once after the first TTS chunk is queued so summarization
                # is detached from (and never on) the hot path.
                _deferred_fired = False
                # V-08: raw (unvalidated) attrs for the CURRENT in-progress
                # segment's pending [DELIVERY ...] tag, or None. Built into a
                # real SegmentIntent (via build_segment_intent) only once the
                # segment's final text is known at flush time, then reset —
                # each new segment starts with no pending tag. Fully inert
                # (stays None forever) whenever VOICE_ENABLE_LLM_HUMANIZATION
                # is off, since the LLM is never asked to emit the tag.
                _pending_delivery_attrs: dict | None = None
                _llm_humanization_on = bool(
                    getattr(settings, "VOICE_ENABLE_LLM_HUMANIZATION", False)
                )

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
                    # V-08 defense-in-depth: catches a still-INCOMPLETE (never
                    # closed) [DELIVERY ...] fragment left over once the LLM
                    # stream has fully ended (e.g. max_tokens truncation) —
                    # consume_delivery_tag() correctly leaves such a fragment
                    # alone mid-stream so it can keep accumulating, but once
                    # streaming is done nothing will ever close it. Also
                    # catches any complete tag this call site didn't already
                    # go through consume_delivery_tag for.
                    out = strip_delivery_tags(out)
                    return out

                def _find_flush_index(buf: str):
                    return find_sentence_flush_index(
                        buf, self._h.TTS_FLUSH_MIN_WORDS, self._h.TTS_FLUSH_MAX_WORDS
                    )

                def _find_time_flush_index(buf: str):
                    return find_time_flush_index(
                        buf, max(self._h.TTS_FLUSH_MIN_WORDS, 5), 8
                    )

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

                    # V-08: pull a complete pending [DELIVERY ...] tag out of
                    # the buffer as soon as it fully arrives, BEFORE any
                    # flush-boundary search, so its bracket text is never
                    # counted toward word/sentence boundaries and never
                    # spoken. A partial (unclosed) tag is left untouched to
                    # keep accumulating across more streamed tokens.
                    if _llm_humanization_on:
                        _attrs, tts_buffer = consume_delivery_tag(tts_buffer)
                        if _attrs is not None:
                            _pending_delivery_attrs = _attrs

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
                        # V-08 defense-in-depth: even though consume_delivery_tag
                        # above already drains every COMPLETE tag before this
                        # flush runs, strip again here in case a flush boundary
                        # ever lands inside a not-yet-recognized fragment — the
                        # caller must never hear literal "[DELIVERY ...]" text.
                        to_speak = strip_delivery_tags(to_speak)
                        if to_speak:
                            chunk_counter += 1
                            # V-08: attach this segment's pending delivery
                            # intent (if any), then reset — each NEW segment
                            # starts with no pending tag of its own.
                            _segment_intent = None
                            if _pending_delivery_attrs is not None:
                                _segment_intent = segment_intent_from_tag_attrs(
                                    to_speak, _pending_delivery_attrs
                                )
                                _pending_delivery_attrs = None
                            await self._h._tts_pipeline.queue_tts(
                                {
                                    "text": to_speak,
                                    "chunk_id": chunk_counter,
                                    "use_ssml": self._h._use_ssml,
                                    "is_final": False,
                                    "end_call_after": False,
                                    "_llm_delivery_intent": _segment_intent,
                                }
                            )
                            _vm = getattr(self._h, "_voice_metrics", None)
                            if _vm:
                                _vm.mark_first_tts_queued()
                            # V-07: fire background summarization once, after first TTS chunk.
                            if not _deferred_fired:
                                _deferred_fired = True
                                asyncio.create_task(
                                    self._deferred_conversation_memory_update(user_text)
                                )
                            last_flush_ts = now_ts

                # Flush any remaining buffer as final
                full_accum = response_accum.strip()
                end_call_after = end_call_after or ("[END_CALL]" in full_accum)
                if _transfer_re.search(full_accum):
                    transfer_after = True
                    end_call_after = False
                final_text = _strip_control_tokens(tts_buffer).strip()
                if final_text and not self._h._tts_cancel.is_set() and self._h._tts_pipeline:
                    chunk_counter += 1
                    _final_segment_intent = None
                    if _pending_delivery_attrs is not None:
                        _final_segment_intent = segment_intent_from_tag_attrs(
                            final_text, _pending_delivery_attrs
                        )
                        _pending_delivery_attrs = None
                    await self._h._tts_pipeline.queue_tts(
                        {
                            "text": final_text,
                            "chunk_id": chunk_counter,
                            "use_ssml": self._h._use_ssml,
                            "is_final": True,
                            "end_call_after": end_call_after and not transfer_after,
                            "transfer_after": transfer_after,
                            "_llm_delivery_intent": _final_segment_intent,
                        }
                    )
                    _vm = getattr(self._h, "_voice_metrics", None)
                    if _vm:
                        _vm.mark_first_tts_queued()
                    # V-07: fire background summarization if it hasn't fired already
                    # (covers short responses that bypass the mid-stream sentence-boundary path).
                    if not _deferred_fired:
                        _deferred_fired = True
                        asyncio.create_task(
                            self._deferred_conversation_memory_update(user_text)
                        )
                elif transfer_after and not self._h._tts_cancel.is_set() and self._h._tts_pipeline:
                    chunk_counter += 1
                    await self._h._tts_pipeline.queue_tts(
                        {
                            "text": random.choice(_TRANSFER_PHRASES),
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
                    # V-07: fire for transfer-phrase path too.
                    if not _deferred_fired:
                        _deferred_fired = True
                        asyncio.create_task(
                            self._deferred_conversation_memory_update(user_text)
                        )
                if self._h.call_session:
                    try:
                        _estimated_tokens = int(
                            (len(system_prompt or "") + len(response_accum or "")) / 3.8
                        )
                        if _estimated_tokens > 0:
                            asyncio.create_task(
                                token_budget_service.record_daily_tokens(
                                    self._h.call_session.tenant_id, _estimated_tokens
                                )
                            )
                    except Exception as exc:
                        logger.debug("Token budget recording failed: %s", exc)
                # V-08: strip any [DELIVERY ...] tags from the RAW accumulated
                # text before it's used for the transcript / token-budget
                # estimate above — already-spoken tags were incrementally
                # removed from tts_buffer per-segment above, but
                # response_accum keeps the untouched raw stream for these
                # other consumers.
                return strip_delivery_tags(response_accum)

            final_text = ""
            try:
                final_text = await try_stream(llm_service, model_name, api_key_override=api_key)
                if _attempted_primary:
                    await llm_circuit_breaker.record_success(_primary_provider)
                logger.debug(
                    "[LLM] response received: chars=%s chunks_queued=%s",
                    len(final_text or ""), chunk_counter,
                )
            except Exception as e:
                logger.error("LLM streaming failed: %s", e, exc_info=True)
                if _attempted_primary:
                    await llm_circuit_breaker.record_failure(_primary_provider, e)
                # A-05: without this, an LLM failure here (primary or the
                # breaker's own secondary swap) leaves the caller with pure
                # silence — mirror bidirectional_stream.py's ultimate
                # fallback and speak a canned message instead of going quiet.
                fallback_text = (
                    getattr(settings, "VOICE_LLM_FALLBACK_MESSAGE", None)
                    or "I am sorry, I did not catch that. Could you please repeat that?"
                )
                if not self._h._tts_cancel.is_set() and self._h._tts_pipeline:
                    chunk_counter += 1
                    await self._h._tts_pipeline.queue_tts(
                        {
                            "text": fallback_text,
                            "chunk_id": chunk_counter,
                            "use_ssml": self._h._use_ssml,
                            "is_final": True,
                        }
                    )
                    _vm = getattr(self._h, "_voice_metrics", None)
                    if _vm:
                        _vm.mark_first_tts_queued()
                final_text = fallback_text

            if final_text:
                transcript_text = re.sub(
                    r"\[\s*TRANSFER_CALL\s*\]", "", final_text, flags=re.IGNORECASE
                ).replace("[END_CALL]", "").strip()
                if transcript_text:
                    await self._h._add_to_transcript("agent", transcript_text, "agent_response")

        except Exception as e:
            logger.error("Error in generate_and_stream_response: %s", e, exc_info=True)

    # ---- V-07 History Summarization Pipeline --------------------------------

    async def _deferred_conversation_memory_update(self, user_text: str = "") -> None:
        """
        Non-blocking hook for background history summarization.  Must be called via
        ``asyncio.create_task(...)`` after the first TTS chunk has been queued — it is
        always out-of-band and never adds latency to the STT -> LLM -> TTS hot path.

        Mirrors BidirectionalStreamHandler._deferred_conversation_memory_update exactly
        but uses the conversation_history_cache on the handler (self._h) rather than
        a locally owned cache, and tracks summarization state on self.
        """
        try:
            from app.core.config import settings as _settings

            _summarization_enabled = getattr(_settings, "VOICE_HISTORY_SUMMARIZATION_ENABLED", True)
            if not _summarization_enabled:
                return

            # Resolve the history cache from the handler (shared mutable list).
            _cache: list[tuple[str, str]] = getattr(self._h, "_conversation_history_cache", [])
            _max_msgs: int = getattr(self._h, "HISTORY_MAX_MESSAGES", 50)
            _chunk_size: int = getattr(_settings, "VOICE_HISTORY_SUMMARY_CHUNK_SIZE", 10)
            _total_turns: int = len(_cache)

            if _total_turns <= _max_msgs:
                return

            _drop_boundary: int = _total_turns - _max_msgs
            _unsummarized_count: int = _drop_boundary - self._last_summarized_turn_index
            if _unsummarized_count < _chunk_size:
                return

            _turns_to_compress: list[tuple[str, str]] = list(
                _cache[self._last_summarized_turn_index : _drop_boundary]
            )
            _current_summary: str = self._history_summary
            _new_index: int = _drop_boundary

            async def _background_summarize() -> None:
                try:
                    from app.services.history_summarization_service import compress_history

                    updated_summary = await compress_history(
                        existing_summary=_current_summary,
                        new_turns=_turns_to_compress,
                    )
                    self._history_summary = updated_summary
                    self._last_summarized_turn_index = _new_index
                    logger.debug(
                        "[V-07] Orchestrator history summary updated: index=%d summary_len=%d",
                        _new_index,
                        len(updated_summary),
                    )
                except Exception as _bg_exc:  # pragma: no cover - defensive
                    logger.debug("[V-07] Orchestrator background summarization failed: %s", _bg_exc)

            asyncio.create_task(_background_summarize())

        except Exception as _exc:
            logger.debug("[V-07] Orchestrator summarization scheduling failed: %s", _exc)

    # ---- High-level entrypoint ----------------------------------------------

    async def on_user_speech(
        self,
        text: str,
        is_final: bool,
        audio_stats: Dict[str, Any] | None = None,
        timestamps: Dict[str, Any] | None = None,
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

