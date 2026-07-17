"""
Vertex AI Gemini LLM service for real-time voice calls.

Uses google-cloud-aiplatform SDK + Application Default Credentials (ADC).
Never requires a per-model api_key — auth is via GOOGLE_APPLICATION_CREDENTIALS.
"""

from __future__ import annotations

import asyncio
import enum
import threading
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from app.core.config import settings
from app.core.logger import logger
from app.voice.llm_prompt_builder import build_vertex_contents

# Lazily imported so tests can patch before import resolution.
_vertex_init_lock = threading.Lock()
_vertexai_initialized = False


def _ensure_vertex_init() -> None:
    global _vertexai_initialized
    if _vertexai_initialized:
        return
    with _vertex_init_lock:
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


def _build_calendly_tool():
    """
    Gemini FunctionDeclarations for the Calendly booking flow:
      - check_availability(date): list bookable slots
      - book_appointment(slot, attendee_email): schedule on Calendly
    Lazily imported so environments without google-cloud-aiplatform installed
    can still import this module (matches the rest of this file's pattern).
    """
    from vertexai.generative_models import FunctionDeclaration, Tool

    check_availability = FunctionDeclaration(
        name="check_availability",
        description=(
            "Check available appointment slots on the connected Calendly calendar. "
            "Slot length is fixed by the connected Calendly event type — it cannot "
            "be requested per-call."
        ),
        parameters={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date to check availability for, as YYYY-MM-DD (or 'today'/'tomorrow').",
                },
            },
            "required": ["date"],
        },
    )
    book_appointment = FunctionDeclaration(
        name="book_appointment",
        description="Schedule an appointment on Calendly for a previously offered slot.",
        parameters={
            "type": "object",
            "properties": {
                "slot": {
                    "type": "string",
                    "description": "The chosen slot start time, ISO-8601 (UTC).",
                },
                "attendee_email": {
                    "type": "string",
                    "description": "The caller's email address to send the Calendly confirmation to.",
                },
            },
            "required": ["slot", "attendee_email"],
        },
    )
    return Tool(function_declarations=[check_availability, book_appointment])


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

        model = GenerativeModel(
            model_name=model_name,
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
            response_stream = model.generate_content_async(
                contents,
                generation_config=generation_config,
                stream=True,
            )
            async for response in response_stream:
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

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        conversation_history: list[tuple[str, str]] | None = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 200,
        tool_executor: Callable[[str, dict], Awaitable[dict]],
    ) -> str:
        """
        One conversational turn with Gemini native function calling enabled
        (Calendly check_availability / book_appointment tools).

        If the model returns a function_call part, ``tool_executor(name, args)``
        is awaited to resolve it, the result is fed back as a
        ``Part.from_function_response``, and generation resumes so the model
        can produce the final spoken reply. Returns the final text (never a
        raw function-call JSON blob).
        """
        try:
            _ensure_vertex_init()
        except Exception as exc:
            raise VertexLlmError(f"Vertex init failed: {exc}", VertexLlmErrorType.UNKNOWN) from exc

        try:
            from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
        except ImportError as exc:
            raise VertexLlmError(
                "google-cloud-aiplatform is not installed. Add it to requirements.txt.",
                VertexLlmErrorType.UNKNOWN,
            ) from exc

        model_kwargs: dict = {"tools": [_build_calendly_tool()]}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt

        model = GenerativeModel(model_name=model_name, **model_kwargs)

        max_turns = getattr(settings, "VOICE_LLM_HISTORY_MAX_TURNS", 20)
        contents = build_vertex_contents(conversation_history, prompt, None, max_turns=max_turns)
        generation_config = GenerationConfig(
            temperature=float(temperature), max_output_tokens=int(max_tokens)
        )

        try:
            response = await model.generate_content_async(
                contents, generation_config=generation_config
            )

            candidate = response.candidates[0] if response.candidates else None
            parts = list(candidate.content.parts) if candidate and candidate.content else []
            function_call_part = next((p for p in parts if getattr(p, "function_call", None)), None)

            if function_call_part is None:
                return (response.text or "").strip()

            fc = function_call_part.function_call
            fc_name = fc.name
            fc_args = dict(fc.args) if fc.args else {}
            logger.info("[VertexGemini] function_call name=%s args=%s", fc_name, fc_args)

            tool_result = await tool_executor(fc_name, fc_args)

            contents.append(candidate.content)
            contents.append(
                Content(
                    role="function",
                    parts=[Part.from_function_response(name=fc_name, response=tool_result)],
                )
            )

            follow_up = await model.generate_content_async(
                contents, generation_config=generation_config
            )
            return (follow_up.text or "").strip()
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
            model_name=model_name,
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
