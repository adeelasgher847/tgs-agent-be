"""
LLMStreamManager: Streaming LLM with early speculation and SSML enrichment.

Replaces ConversationOrchestrator (V1 queue-based) with pure async/await.

Key capabilities:
- Early speculation: Start LLM on 3+ word interim (saves ~250ms)
- Cancellation: CancellationToken checked per-token — stops instantly on barge-in
- Re-run: If final transcript diverges from interim, cancel & re-run
- Chunking: Accumulate tokens into 2-12 word batches for TTS
- SSML: Enrich chunks with ElevenLabs audio tags when supported
- Provider routing: Groq → OpenAI → Gemini by priority
"""

import asyncio
import json
import logging
import random
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logger import logger as app_logger
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service
from app.services.groq_service import groq_service
from app.utils.eleven_tts_text import (
    build_elevenlabs_audio_tag_prompt_block,
    get_elevenlabs_voice_prompt_rule_lines,
    supports_elevenlabs_audio_tags,
    prepare_tts_text_for_provider,
    apply_elevenlabs_breathing_fallback,
    strip_eleven_v3_style_tags_for_non_eleven_tts,
)
from app.voice.cancellation import CancellationToken

if TYPE_CHECKING:
    from app.voice.orchestrator import VoiceOrchestrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quick acknowledgement helpers
# ---------------------------------------------------------------------------

_QUICK_ACKS = [
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

_ACK_SKIP_PHRASES = frozenset({
    "help", "emergency", "urgent", "problem", "issue", "sad", "angry",
    "please help", "asap", "critical", "wrong", "broken", "not working", "complaint",
})


def _should_quick_ack(text: str, min_words: int, skip_phrases: frozenset) -> bool:
    """Check if a quick ack is eligible for this user text."""
    words = text.strip().split()
    if len(words) < min_words:
        return False
    lower = text.lower()
    return not any(phrase in lower for phrase in skip_phrases)


# ---------------------------------------------------------------------------
# Chunk flushing helpers
# ---------------------------------------------------------------------------

def _find_flush_index(buf: str, min_words: int, max_words: int) -> Optional[int]:
    """
    Find the best flush point in a token buffer.

    Priority:
    1. Sentence boundary (. ! ?) with >= min_words before it
    2. Soft boundary (, ; :) when buffer >= max_words
    """
    if not buf:
        return None

    # Sentence boundary
    last_boundary = None
    for m in re.finditer(r"([.!?])(\s+|$)", buf):
        last_boundary = m.end(1)

    if last_boundary is not None:
        prefix = buf[:last_boundary].strip()
        if len(prefix.split()) >= min_words:
            return last_boundary

    # Soft boundary on max words
    words = buf.split()
    if len(words) >= max_words:
        last_soft = None
        for m in re.finditer(r"([,;:])(\s+|$)", buf):
            last_soft = m.end(1)
        if last_soft is not None:
            prefix = buf[:last_soft].strip()
            if len(prefix.split()) >= min_words:
                return last_soft

    return None


def _find_time_flush_index(buf: str, min_words: int) -> Optional[int]:
    """Fallback: flush up to 8 words after 0.8s timeout."""
    if not buf:
        return None
    words = buf.split()
    if len(words) < max(min_words, 5):
        return None
    target = min(8, len(words))
    m = re.match(rf"^(?:\S+\s+){{{target - 1}}}\S+", buf)
    return m.end() if m else None


def _strip_control_tokens(text: str) -> str:
    """Remove [END_CALL] and [OUTCOME:...] markers before sending to TTS."""
    text = text.replace("[END_CALL]", "")
    return re.sub(r"\[OUTCOME:[^\]]+\]", "", text)


# ---------------------------------------------------------------------------
# LLMStreamManager
# ---------------------------------------------------------------------------


class LLMStreamManager:
    """
    Manages LLM provider connection, streaming, early speculation and SSML enrichment.

    Call flow:
      orchestrator.on_stt_interim()
        → stream_speculative(partial_text)  [non-blocking task]
          → tokens accumulate → chunks → orchestrator.on_llm_chunk()
      orchestrator.on_stt_final()
        → finalize_and_rerun(final_text)   [if diverged from interim]

    Barge-in path:
      cancellation_token.cancel_all()
        → token loop checks is_cancelled() → breaks immediately
    """

    def __init__(
        self,
        call_id: str,
        agent_config: Dict[str, Any],
        orchestrator: "VoiceOrchestrator",
    ) -> None:
        self.call_id = call_id
        self.agent_config = agent_config
        self.orchestrator = orchestrator

        # Provider selection (resolved at init time — inline, no ProviderSelector)
        self._llm_service, self._model_name, self._api_key, self._temperature, self._max_tokens = (
            self._resolve_provider(agent_config)
        )

        # Token accumulation
        self._token_buffer: str = ""
        self._chunk_counter: int = 0
        self._last_flush_ts: float = 0.0
        self._clean_response_accum: str = ""

        # Speculation tracking
        self.has_speculation: bool = False
        self._speculative_task: Optional[asyncio.Task] = None
        self._interim_text_at_speculation: str = ""

        # TTS provider metadata for SSML enrichment
        self._tts_provider_slug: str = ""
        self._use_elevenlabs_audio_tags: bool = False
        self._elevenlabs_audio_tag_block: str = ""
        self._output_plain_text_rule: str = ""
        self._no_ssml_rule: str = ""
        self._no_ssml_rule_base: str = ""

        # Flush tunables
        self._flush_min_words: int = settings.VOICE_TTS_FLUSH_MIN_WORDS
        self._flush_max_words: int = settings.VOICE_TTS_FLUSH_MAX_WORDS

        self._configure_tts_rules()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure_tts_provider(self, tts_provider_slug: str) -> None:
        """Update TTS provider slug and refresh SSML rules."""
        self._tts_provider_slug = tts_provider_slug
        self._configure_tts_rules()

    async def stream_speculative(
        self,
        partial_text: str,
        cancellation_token: CancellationToken,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str,
        is_greeting: bool = False,
    ) -> None:
        """
        Start LLM streaming on interim transcript (speculative — may be cancelled).

        Non-blocking: must be called via asyncio.create_task().

        Args:
            partial_text: Interim STT transcript (3+ words).
            cancellation_token: Shared token — cancelled on barge-in.
            conversation_history: Current message history for context.
            system_prompt: Pre-built system prompt from orchestrator.
            is_greeting: If True, skip LLM and stream greeting directly.
        """
        self.has_speculation = True
        self._interim_text_at_speculation = partial_text
        self._token_buffer = ""
        self._chunk_counter = 0
        self._last_flush_ts = time.perf_counter()
        self._clean_response_accum = ""

        try:
            if is_greeting:
                await self._stream_greeting(partial_text, cancellation_token)
                return

            await self._run_llm_stream(
                user_text=partial_text,
                system_prompt=system_prompt,
                cancellation_token=cancellation_token,
            )
        except asyncio.CancelledError:
            logger.debug(f"[{self.call_id}] LLM speculation cancelled (CancelledError)")
        except Exception as e:
            logger.error(f"[{self.call_id}] LLM speculation error: {e}", exc_info=True)
        finally:
            self.has_speculation = False
            self._speculative_task = None

    async def finalize_and_rerun(
        self,
        final_text: str,
        cancellation_token: CancellationToken,
        conversation_history: List[Dict[str, Any]],
        system_prompt: str,
    ) -> None:
        """
        Called when STT final diverges from interim text used for speculation.

        Cancels in-flight speculation, resets buffer, re-runs with final text.
        """
        # Cancel ongoing speculation
        if self._speculative_task and not self._speculative_task.done():
            self._speculative_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._speculative_task), timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        self._token_buffer = ""
        self._chunk_counter = 0
        self._clean_response_accum = ""
        self.has_speculation = False

        # Reset cancellation token for re-run (barge-in cleared)
        # Note: orchestrator creates fresh token for each turn — we just run
        logger.info(f"[{self.call_id}] LLM re-running with final: '{final_text[:30]}...'")

        self._speculative_task = asyncio.create_task(
            self.stream_speculative(
                partial_text=final_text,
                cancellation_token=cancellation_token,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
            )
        )
        await cancellation_token.register_task(self._speculative_task)

    async def send_quick_ack(self) -> None:
        """
        Emit a quick acknowledgement phrase immediately (before LLM finishes).

        Skips emotional/serious content. Probability-gated (38% by default).
        """
        user_text = self._interim_text_at_speculation
        if not _should_quick_ack(
            user_text,
            min_words=settings.VOICE_QUICK_ACK_MIN_WORDS,
            skip_phrases=_ACK_SKIP_PHRASES,
        ):
            return

        if random.random() >= settings.VOICE_QUICK_ACK_PROBABILITY:
            return

        ack = random.choice(_QUICK_ACKS)
        logger.debug(f"[{self.call_id}] Quick ack: '{ack}'")
        await self.orchestrator.on_llm_chunk(
            text=ack,
            is_final=False,
            end_call_after=False,
            is_quick_ack=True,
        )

    async def stop(self) -> None:
        """Cancel any in-flight speculation and clean up."""
        if self._speculative_task and not self._speculative_task.done():
            self._speculative_task.cancel()
            try:
                await self._speculative_task
            except (asyncio.CancelledError, Exception):
                pass
        self._token_buffer = ""
        self.has_speculation = False
        logger.info(f"[{self.call_id}] LLMStreamManager stopped")

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_provider(
        agent_config: Dict[str, Any],
    ) -> Tuple[Any, str, Optional[str], float, int]:
        """
        Inline provider selection — no separate ProviderSelector module.

        Priority: Groq → OpenAI → Gemini (fastest first).
        """
        service = gemini_service
        model_name = "gemini-1.5-flash"
        api_key: Optional[str] = None
        temperature: float = 0.5
        max_tokens: int = 100

        agent = agent_config.get("agent")
        if not agent:
            return service, model_name, api_key, temperature, max_tokens

        if hasattr(agent, "model") and agent.model:
            model_name = agent.model.model_name or model_name

            if agent.model.api_key:
                try:
                    from app.core.security import decrypt_api_key
                    api_key = decrypt_api_key(agent.model.api_key)
                except Exception as e:
                    logger.warning(f"Failed to decrypt agent API key: {e}")

            if getattr(agent, "agent_temperature", None) is not None:
                temperature = agent.agent_temperature / 100.0
            elif getattr(agent.model, "temperature", None) is not None:
                temperature = agent.model.temperature / 100.0

            if getattr(agent, "agent_max_tokens", None):
                max_tokens = agent.agent_max_tokens
            elif getattr(agent.model, "max_tokens", None):
                max_tokens = agent.model.max_tokens

            # Provider selection
            provider = getattr(agent, "provider", None)
            if provider:
                name = (provider.name or "").lower()
                if "groq" in name:
                    service = groq_service
                elif "openai" in name:
                    service = openai_service
                elif "gemini" in name or "google" in name:
                    service = gemini_service
                else:
                    service = gemini_service

        return service, model_name, api_key, temperature, max_tokens

    # ------------------------------------------------------------------
    # SSML / prompt rule configuration
    # ------------------------------------------------------------------

    def _configure_tts_rules(self) -> None:
        """Build SSML output rules based on TTS provider capabilities."""
        slug = self._tts_provider_slug.lower()
        self._use_elevenlabs_audio_tags = supports_elevenlabs_audio_tags(slug)

        if self._use_elevenlabs_audio_tags:
            (
                self._output_plain_text_rule,
                self._no_ssml_rule_base,
                self._no_ssml_rule,
            ) = get_elevenlabs_voice_prompt_rule_lines()
        else:
            self._output_plain_text_rule = (
                "- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML, XML, or any tags. "
                "Prosody is handled by the system."
            )
            self._no_ssml_rule_base = (
                "4. NO SSML: Do NOT output <speak>, <prosody>, or any XML tags. Plain text only."
            )
            self._no_ssml_rule = "3. NO SSML: Plain text only. No <speak>, <prosody>, or XML."

        self._elevenlabs_audio_tag_block = build_elevenlabs_audio_tag_prompt_block(slug)

    def build_system_prompt(
        self,
        agent_name: str,
        agent_language: str,
        history_text: str,
    ) -> str:
        """
        Build the LLM system prompt for this turn.

        Precedence: agent.system_prompt > agent.model.system_prompt > base prompt
        """
        agent = self.agent_config.get("agent")

        base_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call with a human.

# STYLE & TONE
- VOICE-FIRST: Your output is for Text-to-Speech. Use short, punchy sentences.
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
- CONCISE: Max 20 words per response unless explaining something complex.
- NO ROBOT TALK: Avoid "As an AI" or formal greetings. Use "Hey," "Hi," or "Hello."
{self._output_plain_text_rule}
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

# CRITICAL RULES
1. NO REPETITION: If the history shows you asked a question, move to the next point.
2. HANDLING SILENCE: If the user says something vague, ask a clarifying question.
3. TERMINATION: When the objective is met, say a friendly goodbye and end your response with exactly [END_CALL].
{self._no_ssml_rule_base}

{self._elevenlabs_audio_tag_block}

# GOAL
Continue the conversation based on the history above. Be {agent_name}."""

        if not agent:
            return base_prompt

        if getattr(agent, "system_prompt", None):
            return f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# CUSTOM INSTRUCTIONS
{agent.system_prompt}

# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
{self._output_plain_text_rule}
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

# CRITICAL RULES
1. NO REPETITION: Do not repeat questions already asked. Move to the next point.
2. TERMINATION: When all objectives from your custom instructions are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{self._no_ssml_rule}

{self._elevenlabs_audio_tag_block}

# GOAL
Follow your custom instructions. Continue from the history above. Be {agent_name}."""

        if getattr(agent, "model", None) and getattr(agent.model, "system_prompt", None):
            return f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# MODEL INSTRUCTIONS
{agent.model.system_prompt}

# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use fillers like "uhm," "well," "I see" occasionally.
{self._output_plain_text_rule}

# CONVERSATION STATE
Previous conversation:
{history_text}

# CRITICAL RULES
1. NO REPETITION: Do not repeat questions. Move to the next point.
2. TERMINATION: When all objectives are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{self._no_ssml_rule}

{self._elevenlabs_audio_tag_block}

# GOAL
Follow the model instructions. Continue from the history above. Be {agent_name}."""

        return base_prompt

    # ------------------------------------------------------------------
    # Internal streaming
    # ------------------------------------------------------------------

    async def _stream_greeting(self, greeting_text: str, token: CancellationToken) -> None:
        """Handle greeting path — bypass LLM, stream directly."""
        if not greeting_text or token.is_cancelled():
            return
        await self.orchestrator.on_llm_chunk(
            text=greeting_text,
            is_final=True,
            end_call_after=False,
            is_quick_ack=False,
        )

    async def _run_llm_stream(
        self,
        user_text: str,
        system_prompt: str,
        cancellation_token: CancellationToken,
    ) -> None:
        """
        Core streaming loop: calls provider, accumulates tokens, flushes chunks.

        Checks cancellation_token on every token — breaks immediately on barge-in.
        """
        end_call_after = False
        tts_buffer = ""
        response_accum = ""
        first_tts_chunk = True

        logger.info(
            f"[{self.call_id}] LLM stream start "
            f"({self._llm_service.__class__.__name__}, '{user_text[:20]}...')"
        )

        async for chunk in self._llm_service.stream_text(
            prompt=user_text,
            system_prompt=system_prompt,
            model_name=self._model_name,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=self._api_key,
        ):
            # Check cancellation on every token — instant stop on barge-in
            if cancellation_token.is_cancelled():
                logger.debug(f"[{self.call_id}] LLM stream: cancellation detected — stopping")
                break

            if not chunk:
                continue

            response_accum += chunk
            tts_buffer += chunk

            # Detect control tokens
            if "[END_CALL]" in response_accum:
                end_call_after = True
                tts_buffer = _strip_control_tokens(tts_buffer)

            if "[OUTCOME:" in tts_buffer:
                tts_buffer = _strip_control_tokens(tts_buffer)

            # Quick ack on first meaningful content
            if first_tts_chunk and len(tts_buffer.split()) >= 2:
                await self.send_quick_ack()
                first_tts_chunk = False

            # Try to flush a chunk
            flush_idx = _find_flush_index(
                tts_buffer, self._flush_min_words, self._flush_max_words
            )

            # Time-based fallback flush (0.8s)
            now = time.perf_counter()
            if flush_idx is None and (now - self._last_flush_ts) >= 0.8:
                flush_idx = _find_time_flush_index(tts_buffer, self._flush_min_words)

            if flush_idx is not None and not cancellation_token.is_cancelled():
                to_speak = tts_buffer[:flush_idx].strip()
                tts_buffer = tts_buffer[flush_idx:].lstrip()
                if to_speak:
                    self._chunk_counter += 1
                    self._last_flush_ts = now
                    
                    # Apply ElevenLabs tags & format for provider
                    is_first = (self._chunk_counter == 1)
                    if is_first and self._use_elevenlabs_audio_tags:
                        to_speak = apply_elevenlabs_breathing_fallback(to_speak)

                    # Store transcript-safe text without bracket tags for DB/UI
                    clean_chunk = strip_eleven_v3_style_tags_for_non_eleven_tts(to_speak)
                    if clean_chunk:
                        self._clean_response_accum += clean_chunk + " "

                    to_speak = prepare_tts_text_for_provider(to_speak, self._tts_provider_slug)

                    await self.orchestrator.on_llm_chunk(
                        text=to_speak,
                        is_final=False,
                        end_call_after=False,
                        is_quick_ack=False,
                    )

        # Flush any remaining buffer as final chunk
        final_text = _strip_control_tokens(tts_buffer).strip()
        if final_text and not cancellation_token.is_cancelled():
            self._chunk_counter += 1
            
            # Apply tags & format
            is_first = (self._chunk_counter == 1)
            if is_first and self._use_elevenlabs_audio_tags:
                final_text = apply_elevenlabs_breathing_fallback(final_text)

            clean_final = strip_eleven_v3_style_tags_for_non_eleven_tts(final_text)
            if clean_final:
                self._clean_response_accum += clean_final + " "

            final_text = prepare_tts_text_for_provider(final_text, self._tts_provider_slug)

            await self.orchestrator.on_llm_chunk(
                text=final_text,
                is_final=True,
                end_call_after=end_call_after,
                is_quick_ack=False,
            )

        # Notify orchestrator if END_CALL but no final text
        if end_call_after and not final_text and not cancellation_token.is_cancelled():
            await self.orchestrator.on_llm_end_call()

        logger.info(
            f"[{self.call_id}] LLM stream complete "
            f"({self._chunk_counter} chunks, end_call={end_call_after})"
        )

    def get_agent_response_text(self) -> str:
        """Return the clean, transcript-safe agent response text."""
        return self._clean_response_accum.strip()
