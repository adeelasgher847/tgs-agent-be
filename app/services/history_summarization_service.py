"""
app/services/history_summarization_service.py — V-07 History Summarization Pipeline

Compresses dropped conversation turns into a rolling summary using a fast mini-LLM
(gpt-4o-mini with gemini-2.5-flash fallback) so that early-call context (caller name,
problem, location, constraints) is never silently lost when the bounded history window
slides forward.

**Zero-latency guarantee**: this module is ONLY called from
``_deferred_conversation_memory_update``, which runs completely out-of-band after the
first TTS chunk has already been queued.  It MUST NEVER be awaited on the hot path
(STT -> LLM -> TTS).
"""

from __future__ import annotations

import asyncio
from typing import List, Tuple

from app.core.logger import logger

# ---------------------------------------------------------------------------
# Public constants (imported by handler modules to avoid magic numbers)
# ---------------------------------------------------------------------------

#: Mini-LLM used for turn compression — fast, cheap, accurate enough for extraction.
SUMMARIZATION_MODEL_PRIMARY = "gpt-4o-mini"
SUMMARIZATION_MODEL_FALLBACK = "gemini-2.5-flash"

#: Hard timeout for a single summarization call (seconds).
SUMMARIZATION_TIMEOUT_SEC = 4.0

#: Maximum tokens for the summary output — kept tight to enforce conciseness.
SUMMARIZATION_MAX_TOKENS = 200

#: Temperature: low for factual extraction, not creative generation.
SUMMARIZATION_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a concise call-notes extractor for a voice AI agent. "
    "Given a rolling summary of earlier conversation turns (may be empty) and a batch "
    "of new dropped turns, extract and merge the KEY FACTUAL POINTS that a voice agent "
    "must remember to avoid re-asking questions already answered. "
    "Focus on: caller name, company, location, requested service, key dates or deadlines, "
    "slot/appointment confirmations, pricing constraints, special requirements, and any "
    "explicit preferences or refusals. "
    "Output ONLY the updated summary as concise bullet points or 1-2 dense paragraphs "
    "(<100 words). Do NOT include commentary, headers, or anything except the summary text."
)


def _build_user_prompt(
    existing_summary: str,
    new_turns: List[Tuple[str, str]],
) -> str:
    """Compose the user-facing prompt for the mini-LLM."""
    parts: List[str] = []

    if existing_summary.strip():
        parts.append(f"EXISTING SUMMARY:\n{existing_summary.strip()}")
    else:
        parts.append("EXISTING SUMMARY:\n(none)")

    if new_turns:
        turns_text = "\n".join(
            f"{role.capitalize()}: {content}" for role, content in new_turns
        )
        parts.append(f"NEW DROPPED TURNS TO INCORPORATE:\n{turns_text}")
    else:
        parts.append("NEW DROPPED TURNS TO INCORPORATE:\n(none)")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Core compress function
# ---------------------------------------------------------------------------


async def compress_history(
    existing_summary: str,
    new_turns: List[Tuple[str, str]],
) -> str:
    """
    Compress ``new_turns`` into the running ``existing_summary`` using a fast mini-LLM.

    Args:
        existing_summary: The current rolling summary for this call (may be empty string).
        new_turns: List of ``(role, content)`` tuples that have been dropped from the
                   active sliding window and have not yet been summarized.

    Returns:
        Updated summary string.  On any failure (LLM error, timeout, empty response),
        returns ``existing_summary`` unchanged — this function is guaranteed never to raise.

    **Hot-path safety**: this is an ``async`` coroutine but is always scheduled via
    ``asyncio.create_task(...)`` from ``_deferred_conversation_memory_update`` — it is
    never awaited directly on the STT->LLM->TTS path.
    """
    if not new_turns:
        # Nothing to compress; return existing summary as-is (no LLM call).
        return existing_summary

    user_prompt = _build_user_prompt(existing_summary, new_turns)

    # Try primary provider (OpenAI gpt-4o-mini), then fallback (Gemini).
    result = await _try_openai(user_prompt)
    if result is None:
        result = await _try_gemini(user_prompt)

    if result is None:
        logger.debug(
            "History summarization: both primary and fallback LLMs failed; "
            "preserving existing summary (%d chars).",
            len(existing_summary),
        )
        return existing_summary

    logger.debug(
        "History summarization: compressed %d new turns -> %d-char summary.",
        len(new_turns),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Provider implementations — each returns the summary string or None on failure
# ---------------------------------------------------------------------------


async def _try_openai(user_prompt: str) -> str | None:
    """Attempt summarization via OpenAI gpt-4o-mini with a strict timeout."""
    try:
        from app.core.config import settings

        api_key = getattr(settings, "OPENAI_API_KEY", "") or ""
        if not api_key:
            return None

        # Import lazily to avoid module-level dep on openai being configured.
        from openai import AsyncOpenAI  # type: ignore[import]

        client = AsyncOpenAI(api_key=api_key)

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=SUMMARIZATION_MODEL_PRIMARY,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=SUMMARIZATION_MAX_TOKENS,
                temperature=SUMMARIZATION_TEMPERATURE,
            ),
            timeout=SUMMARIZATION_TIMEOUT_SEC,
        )

        content = (response.choices[0].message.content or "").strip()
        return content if content else None

    except asyncio.TimeoutError:
        logger.debug(
            "History summarization (OpenAI): timed out after %.1fs.",
            SUMMARIZATION_TIMEOUT_SEC,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("History summarization (OpenAI): %s", exc)
        return None


async def _try_gemini(user_prompt: str) -> str | None:
    """Attempt summarization via Gemini flash as fallback using the google-genai SDK."""
    try:
        from app.core.config import settings

        api_key = getattr(settings, "GEMINI_API_KEY", "") or ""
        if not api_key:
            return None

        from google import genai  # type: ignore[import]
        from google.genai import types as genai_types  # type: ignore[import]

        client = genai.Client(api_key=api_key)

        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=SUMMARIZATION_MODEL_FALLBACK,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    max_output_tokens=SUMMARIZATION_MAX_TOKENS,
                    temperature=SUMMARIZATION_TEMPERATURE,
                ),
            ),
            timeout=SUMMARIZATION_TIMEOUT_SEC,
        )

        content = (getattr(response, "text", None) or "").strip()
        return content if content else None

    except asyncio.TimeoutError:
        logger.debug(
            "History summarization (Gemini): timed out after %.1fs.",
            SUMMARIZATION_TIMEOUT_SEC,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("History summarization (Gemini): %s", exc)
        return None
