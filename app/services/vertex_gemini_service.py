"""
Vertex AI Gemini LLM service for real-time voice calls.

Uses google-cloud-aiplatform SDK + Application Default Credentials (ADC).
Never requires a per-model api_key — auth is via GOOGLE_APPLICATION_CREDENTIALS.
"""

from __future__ import annotations

import asyncio
import enum
import time
from typing import Any, AsyncIterator

from app.core.config import settings
from app.core.logger import logger
from app.voice.llm_prompt_builder import build_vertex_contents

# Lazily imported so tests can patch before import resolution.
_vertexai_initialized = False


def _ensure_vertex_init() -> None:
    global _vertexai_initialized
    if _vertexai_initialized:
        return
    import vertexai

    project = settings.GOOGLE_CLOUD_PROJECT_ID or settings.GCP_PROJECT_ID
    location = settings.VERTEX_AI_LOCATION
    if not project:
        raise RuntimeError(
            "Vertex AI requires GOOGLE_CLOUD_PROJECT_ID or GCP_PROJECT_ID in config."
        )
    vertexai.init(project=project, location=location)
    _vertexai_initialized = True


class VertexLlmErrorType(str, enum.Enum):
    QUOTA = "quota"
    TIMEOUT = "timeout"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


class VertexLlmError(Exception):
    def __init__(self, message: str, error_type: VertexLlmErrorType = VertexLlmErrorType.UNKNOWN) -> None:
        super().__init__(message)
        self.error_type = error_type


def _classify_vertex_error(exc: Exception) -> VertexLlmError:
    name = type(exc).__name__
    msg = str(exc)

    # Match against google.api_core exception types when the package is available.
    # Catch TypeError as a defensive measure: google.api_core's custom metaclass
    # (_GoogleAPICallErrorMeta) can raise TypeError on isinstance() in some
    # Python/test-runner environments when the class identity is ambiguous.
    try:
        from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted

        if isinstance(exc, ResourceExhausted):
            return VertexLlmError(f"Vertex quota exceeded: {msg}", VertexLlmErrorType.QUOTA)
        if isinstance(exc, DeadlineExceeded):
            return VertexLlmError(f"Vertex deadline exceeded: {msg}", VertexLlmErrorType.TIMEOUT)
    except (ImportError, TypeError):
        pass

    # Fallback: classify by class name / message text
    if "ResourceExhausted" in name or "quota" in msg.lower():
        return VertexLlmError(f"Vertex quota: {msg}", VertexLlmErrorType.QUOTA)
    if "DeadlineExceeded" in name or "timeout" in msg.lower() or "deadline" in msg.lower():
        return VertexLlmError(f"Vertex timeout: {msg}", VertexLlmErrorType.TIMEOUT)
    if "blocked" in msg.lower() or "safety" in msg.lower() or "filter" in msg.lower():
        return VertexLlmError(f"Vertex content filter: {msg}", VertexLlmErrorType.CONTENT_FILTER)
    return VertexLlmError(f"Vertex error: {msg}", VertexLlmErrorType.UNKNOWN)


class VertexGeminiService:
    """
    Singleton service that streams Gemini 2.5 Flash responses via Vertex AI.

    Auth: Application Default Credentials — never reads model.api_key.
    """

    async def stream_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        conversation_history: list[tuple[str, str]] | None = None,
        kb_context: str | None = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 100,
        cancel_event: asyncio.Event | None = None,
        # Ignored — Vertex uses ADC, not api_key. Accepted for interface compatibility.
        api_key: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Async generator that yields text chunks from Vertex Gemini.

        Checks cancel_event each iteration so barge-in stops the stream immediately.
        Raises VertexLlmError on quota/timeout/filter failures (caller maps to fallback).
        """
        try:
            _ensure_vertex_init()
        except Exception as exc:
            raise VertexLlmError(f"Vertex init failed: {exc}", VertexLlmErrorType.UNKNOWN) from exc

        try:
            from vertexai.generative_models import GenerativeModel, GenerationConfig
        except ImportError as exc:
            raise VertexLlmError(
                "google-cloud-aiplatform is not installed. Add it to requirements.txt.",
                VertexLlmErrorType.UNKNOWN,
            ) from exc

        # Build model with system_instruction (never log full prompt).
        model_kwargs: dict = {}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt

        full_model_name = f"publishers/google/models/{model_name}"

        model = GenerativeModel(
            model_name=full_model_name,
            **model_kwargs,
        )

        max_turns = getattr(settings, "VOICE_LLM_HISTORY_MAX_TURNS", 20)
        contents = build_vertex_contents(
            conversation_history, prompt, kb_context, max_turns=max_turns
        )

        generation_config = GenerationConfig(
            temperature=float(temperature),
            max_output_tokens=int(max_tokens),
        )

        # Use the async SDK method so chunks are yielded as they arrive without
        # blocking the event loop.  The sync generate_content() path runs its
        # entire HTTP round-trip inside asyncio.to_thread and only returns after
        # ALL tokens are buffered — effectively disabling streaming.
        try:
            async for response in await model.generate_content_async(
                contents,
                generation_config=generation_config,
                stream=True,
            ):
                # Honour barge-in / interruption cancel
                if cancel_event is not None and cancel_event.is_set():
                    logger.debug("[VertexGemini] cancel_event set — stopping stream")
                    break

                try:
                    text = response.text
                except Exception:
                    # Candidate may be blocked or empty (content filter)
                    try:
                        candidates = getattr(response, "candidates", [])
                        if candidates:
                            finish = getattr(candidates[0], "finish_reason", None)
                            if finish and str(finish) not in ("STOP", "MAX_TOKENS", "1", "2"):
                                raise VertexLlmError(
                                    f"Vertex content blocked: finish_reason={finish}",
                                    VertexLlmErrorType.CONTENT_FILTER,
                                )
                    except VertexLlmError:
                        raise
                    except Exception:
                        pass
                    continue

                if text:
                    yield text
        except VertexLlmError:
            raise
        except Exception as exc:
            raise _classify_vertex_error(exc) from exc

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 100,
        api_key: str | None = None,
        conversation_history: list[tuple[str, str]] | None = None,
        kb_context: str | None = None,
    ) -> dict[str, Any]:
        """
        Non-streaming completion for legacy gather / live-voice paths.
        Same return shape as ``GeminiService.generate_text`` for drop-in use.
        """
        del api_key  # Vertex uses ADC only
        start_time = time.time()
        try:
            _ensure_vertex_init()
        except Exception as exc:
            raise VertexLlmError(f"Vertex init failed: {exc}", VertexLlmErrorType.UNKNOWN) from exc

        try:
            from vertexai.generative_models import GenerativeModel, GenerationConfig
        except ImportError as exc:
            raise VertexLlmError(
                "google-cloud-aiplatform is not installed.",
                VertexLlmErrorType.UNKNOWN,
            ) from exc

        model_kwargs: dict = {}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt

        model = GenerativeModel(
            model_name=f"publishers/google/models/{model_name}",
            **model_kwargs,
        )
        max_turns = getattr(settings, "VOICE_LLM_HISTORY_MAX_TURNS", 20)
        contents = build_vertex_contents(
            conversation_history, prompt, kb_context, max_turns=max_turns
        )
        generation_config = GenerationConfig(
            temperature=float(temperature),
            max_output_tokens=int(max_tokens),
        )

        try:
            response = model.generate_content(
                contents,
                generation_config=generation_config,
            )
            response_text = (response.text or "").strip()
        except Exception as exc:
            raise _classify_vertex_error(exc) from exc

        elapsed = time.time() - start_time
        return {
            "content": response_text,
            "model": model_name,
            "response_time": elapsed,
            "usage": {},
            "finish_reason": "stop",
        }


# Module-level singleton
vertex_gemini_service = VertexGeminiService()
