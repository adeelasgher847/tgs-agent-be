"""
Vertex AI Gemini 2.5 Flash LLM service for the bidirectional voice pipeline.

Auth: GCP Application Default Credentials (ADC) via GOOGLE_APPLICATION_CREDENTIALS
or Workload Identity. No API key — Vertex uses IAM.

Implements the same stream_text() interface as GeminiService so the existing
try_stream() loop in bidirectional_stream.py works without modification.
"""
from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from queue import Empty, Queue
from typing import AsyncIterator, Optional

from app.core.logger import logger

VERTEX_FALLBACK_RESPONSE = "I am sorry, I did not catch that"

# ---------------------------------------------------------------------------
# Typed errors — handler maps these to structured log + fallback
# ---------------------------------------------------------------------------


class VertexQuotaError(Exception):
    """Vertex API quota / rate-limit exceeded (HTTP 429 / RESOURCE_EXHAUSTED)."""


class VertexTimeoutError(Exception):
    """Vertex request deadline exceeded."""


class VertexContentFilterError(Exception):
    """Vertex safety filter blocked the response."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

_SENTINEL = object()


class VertexLlmService:
    """Streaming Gemini 2.5 Flash via Vertex AI Generative AI SDK."""

    def __init__(self) -> None:
        self._initialized = False
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Initialisation (lazy, thread-safe)
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            try:
                import vertexai  # noqa: F401 — imported for side-effects (init)
                from app.core.config import settings  # local import keeps startup fast
                from app.core.google_credentials import ensure_google_application_credentials_env

                creds_path = ensure_google_application_credentials_env()
                if not creds_path:
                    raise RuntimeError(
                        "GOOGLE_APPLICATION_CREDENTIALS is not configured. "
                        "Set a service-account JSON path or inline JSON in .env "
                        "(same credentials as Google TTS)."
                    )

                project = (
                    settings.VERTEX_PROJECT_ID
                    or settings.GOOGLE_CLOUD_PROJECT_ID
                    or settings.GCP_PROJECT_ID
                )
                location = settings.VERTEX_LOCATION
                if not project:
                    raise RuntimeError(
                        "VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT_ID / GCP_PROJECT_ID) "
                        "is not configured. Set it in .env."
                    )
                vertexai.init(project=project, location=location)
                self._initialized = True
                logger.info(
                    "[Vertex] SDK initialised project=%s location=%s", project, location
                )
            except ImportError as exc:
                raise RuntimeError(
                    "google-cloud-aiplatform is not installed. "
                    "Run: pip install google-cloud-aiplatform"
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"Vertex AI init failed: {exc}") from exc

    # ------------------------------------------------------------------
    # stream_text — drop-in replacement for GeminiService.stream_text()
    # ------------------------------------------------------------------

    async def stream_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 100,
        api_key: Optional[str] = None,  # unused — Vertex uses ADC
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """
        Yield text chunks from Vertex Gemini (true server-side streaming).

        Args:
            prompt: The caller's current utterance (user turn).
            system_prompt: Full system prompt — injected as system role, never logged.
            model_name: Vertex model short name (e.g. "gemini-2.5-flash").
            temperature: Sampling temperature (0–1). Voice default 0.3.
            max_tokens: Max output tokens.
            api_key: Ignored — Vertex uses GCP ADC / service account.
            cancel_event: asyncio.Event; set it to abort streaming mid-response.
        """
        self._ensure_init()

        # Safe debug metadata — only len+hash; never the full system prompt content.
        _sp_hash = hashlib.sha256((system_prompt or "").encode()).hexdigest()[:8]
        logger.debug(
            "[Vertex] stream_text model=%s sp_len=%d sp_hash=%s temp=%.2f max_tokens=%d",
            model_name,
            len(system_prompt or ""),
            _sp_hash,
            temperature,
            max_tokens,
        )

        q: Queue = Queue()
        cancel_flag = threading.Event()
        t0 = time.perf_counter()

        def _producer() -> None:
            try:
                from vertexai.generative_models import (
                    Content,
                    GenerationConfig,
                    GenerativeModel,
                    Part,
                )

                model = GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt or None,
                )
                config = GenerationConfig(
                    temperature=float(temperature),
                    max_output_tokens=int(max_tokens),
                )
                contents = [Content(role="user", parts=[Part.from_text(prompt or " ")])]

                response_stream = model.generate_content(
                    contents=contents,
                    generation_config=config,
                    stream=True,
                )
                for chunk in response_stream:
                    if cancel_flag.is_set():
                        break
                    try:
                        text = chunk.text
                        if text:
                            q.put(text)
                    except (AttributeError, ValueError):
                        continue
            except Exception as exc:
                q.put(("__error__", exc))
            finally:
                q.put(_SENTINEL)

        thread = threading.Thread(target=_producer, daemon=True, name="vertex-stream")
        thread.start()

        loop = asyncio.get_running_loop()
        first_token = True
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    cancel_flag.set()
                    logger.debug("[Vertex] stream cancelled by cancel_event")
                    break

                try:
                    item = await loop.run_in_executor(None, lambda: q.get(timeout=30))
                except Empty:
                    cancel_flag.set()
                    raise VertexTimeoutError("Vertex stream timed out after 30s")

                if item is _SENTINEL:
                    break
                if isinstance(item, tuple) and item[0] == "__error__":
                    cancel_flag.set()
                    _classify_and_raise(item[1])

                if first_token:
                    logger.debug(
                        "[Vertex] first token latency=%.3fs", time.perf_counter() - t0
                    )
                    first_token = False

                yield str(item)

        except asyncio.CancelledError:
            cancel_flag.set()
            raise
        finally:
            cancel_flag.set()

    # ------------------------------------------------------------------
    # stream_turn — structured interface (preferred for new callers)
    # ------------------------------------------------------------------

    async def stream_turn(
        self,
        *,
        system_prompt: str,
        conversation_history: list[tuple[str, str]],
        caller_transcript: str,
        kb_context: str = "",
        temperature: float = 0.3,
        max_tokens: int = 100,
        model_name: str = "gemini-2.5-flash",
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """
        Structured streaming with multi-turn Content history.

        Passes system_prompt as system_instruction (no history embedded there)
        and builds a proper multi-turn conversation from conversation_history
        so history is NOT duplicated inside the system blob.

        Role mapping: "client"/"user" → Vertex role "user"; "agent"/"model" → "model".
        """
        self._ensure_init()

        _sp_hash = hashlib.sha256((system_prompt or "").encode()).hexdigest()[:8]
        logger.debug(
            "[Vertex] stream_turn model=%s sp_len=%d sp_hash=%s history_turns=%d temp=%.2f",
            model_name,
            len(system_prompt or ""),
            _sp_hash,
            len(conversation_history),
            temperature,
        )

        # Build current user turn: transcript + optional KB context
        user_message = caller_transcript or " "
        if kb_context:
            user_message = f"{user_message}\n\n[Context]\n{kb_context}"

        q: Queue = Queue()
        cancel_flag = threading.Event()
        t0 = time.perf_counter()

        _history_snapshot = list(conversation_history)  # freeze before async yield

        def _producer() -> None:
            try:
                from vertexai.generative_models import (
                    Content,
                    GenerationConfig,
                    GenerativeModel,
                    Part,
                )

                model = GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt or None,
                )
                config = GenerationConfig(
                    temperature=float(temperature),
                    max_output_tokens=int(max_tokens),
                )

                # Build multi-turn contents: history pairs + current user turn
                def _vertex_role(role: str) -> str:
                    return "model" if role in ("agent", "model", "assistant") else "user"

                contents = []
                for role, content in _history_snapshot:
                    if content:
                        contents.append(
                            Content(
                                role=_vertex_role(role),
                                parts=[Part.from_text(content)],
                            )
                        )
                # Current caller utterance always ends the conversation as a user turn
                contents.append(Content(role="user", parts=[Part.from_text(user_message)]))

                response_stream = model.generate_content(
                    contents=contents,
                    generation_config=config,
                    stream=True,
                )
                for chunk in response_stream:
                    if cancel_flag.is_set():
                        break
                    try:
                        text = chunk.text
                        if text:
                            q.put(text)
                    except (AttributeError, ValueError):
                        continue
            except Exception as exc:
                q.put(("__error__", exc))
            finally:
                q.put(_SENTINEL)

        thread = threading.Thread(target=_producer, daemon=True, name="vertex-turn")
        thread.start()

        loop = asyncio.get_running_loop()
        first_token = True
        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    cancel_flag.set()
                    logger.debug("[Vertex] stream_turn cancelled by cancel_event")
                    break

                try:
                    item = await loop.run_in_executor(None, lambda: q.get(timeout=30))
                except Empty:
                    cancel_flag.set()
                    raise VertexTimeoutError("Vertex stream_turn timed out after 30s")

                if item is _SENTINEL:
                    break
                if isinstance(item, tuple) and item[0] == "__error__":
                    cancel_flag.set()
                    _classify_and_raise(item[1])

                if first_token:
                    logger.debug(
                        "[Vertex] stream_turn first token latency=%.3fs", time.perf_counter() - t0
                    )
                    first_token = False

                yield str(item)

        except asyncio.CancelledError:
            cancel_flag.set()
            raise
        finally:
            cancel_flag.set()


# ---------------------------------------------------------------------------
# Error classifier
# ---------------------------------------------------------------------------


def _classify_and_raise(exc: Exception) -> None:
    """Re-raise exc as a typed VertexError for structured handling by the caller."""
    msg = str(exc).lower()
    if any(k in msg for k in ("quota", "resource exhausted", "429", "ratelimit")):
        raise VertexQuotaError(str(exc)) from exc
    if any(k in msg for k in ("timeout", "deadline", "timed out")):
        raise VertexTimeoutError(str(exc)) from exc
    if any(k in msg for k in ("safety", "blocked", "filter", "harm", "policy")):
        raise VertexContentFilterError(str(exc)) from exc
    raise RuntimeError(str(exc)) from exc


# Shared singleton — import and use this
vertex_llm_service = VertexLlmService()
