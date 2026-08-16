"""
Vertex AI Gemini LLM service for real-time voice calls.

Uses the google-genai SDK (vertexai=True) + Application Default Credentials (ADC).
Never requires a per-model api_key — auth is via GOOGLE_APPLICATION_CREDENTIALS.

Migrated off ``vertexai.generative_models`` (deprecated 2025-06-24, removal date
already passed as of this writing) to ``google.genai`` per Google's official
migration guide.
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
_vertex_client_lock = threading.Lock()
_vertex_clients: dict[str, Any] = {}

# As of 2026-08, Google only serves Gemini 3-family models from Vertex AI's
# ``global`` endpoint — regional endpoints (including our configured
# settings.VERTEX_AI_LOCATION, e.g. "us-central1") return 404 for these
# specific model IDs. This is expected to change as Google rolls out regional
# availability for Gemini 3, so keep this as an easily-editable allowlist
# rather than scattering region special-cases through the call sites.
_GLOBAL_ONLY_MODELS: set[str] = {
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
}


def _location_for_model(model_name: str) -> str:
    """
    Return the Vertex AI location to use for a given model ID: "global" for
    models in _GLOBAL_ONLY_MODELS, otherwise settings.VERTEX_AI_LOCATION.
    Exact-match on model_name (these are exact model IDs, not prefixes).
    """
    if model_name in _GLOBAL_ONLY_MODELS:
        return "global"
    return settings.VERTEX_AI_LOCATION


def _ensure_vertex_client(location: str):
    """
    Lazily build (once per location) and return a shared google.genai Client,
    configured for Vertex AI + ADC. Thread-safe double-checked locking,
    mirroring the old _ensure_vertex_init() singleton pattern, but keyed by
    location so global-only models (see _GLOBAL_ONLY_MODELS) can use the
    "global" endpoint while other models keep using settings.VERTEX_AI_LOCATION.
    """
    existing = _vertex_clients.get(location)
    if existing is not None:
        return existing
    with _vertex_client_lock:
        existing = _vertex_clients.get(location)
        if existing is not None:
            return existing
        from google import genai

        project = settings.GOOGLE_CLOUD_PROJECT_ID or settings.GCP_PROJECT_ID
        if not project:
            raise RuntimeError(
                "Vertex AI requires GOOGLE_CLOUD_PROJECT_ID or GCP_PROJECT_ID in config."
            )
        client = genai.Client(vertexai=True, project=project, location=location)
        _vertex_clients[location] = client
        return client


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

    # Match against google.genai.errors.APIError (base for ClientError/ServerError)
    # when the package is available — mirrors the old google.api_core matching.
    try:
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            status = (getattr(exc, "status", None) or "").upper()
            if code == 429 or status == "RESOURCE_EXHAUSTED":
                return VertexLlmError(f"Vertex quota exceeded: {msg}", VertexLlmErrorType.QUOTA)
            if code in (504, 408) or status == "DEADLINE_EXCEEDED":
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
    Lazily imported so environments without google-genai installed
    can still import this module (matches the rest of this file's pattern).
    """
    from google.genai import types

    check_availability = types.FunctionDeclaration(
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
    book_appointment = types.FunctionDeclaration(
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
    return types.Tool(function_declarations=[check_availability, book_appointment])


def _extract_finish_reason_error(response) -> VertexLlmError | None:
    """Inspect candidate finish_reason for content-filter blocks. Returns None if clean."""
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = getattr(candidates[0], "finish_reason", None)
            if finish and str(finish) not in ("STOP", "MAX_TOKENS", "1", "2"):
                return VertexLlmError(
                    f"Vertex content blocked: finish_reason={finish}",
                    VertexLlmErrorType.CONTENT_FILTER,
                )
    except Exception as exc:
        logger.debug("[VertexGemini] finish_reason inspection failed: %s", exc)
    return None


class VertexGeminiService:
    """
    Singleton service that streams Gemini responses via Vertex AI (google-genai SDK).

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
            client = _ensure_vertex_client(_location_for_model(model_name))
        except Exception as exc:
            raise VertexLlmError(f"Vertex init failed: {exc}", VertexLlmErrorType.UNKNOWN) from exc

        try:
            from google.genai import types
        except ImportError as exc:
            raise VertexLlmError(
                "google-genai is not installed. Add it to requirements.txt.",
                VertexLlmErrorType.UNKNOWN,
            ) from exc

        max_turns = getattr(settings, "VOICE_LLM_HISTORY_MAX_TURNS", 20)
        contents = build_vertex_contents(
            conversation_history, prompt, kb_context, max_turns=max_turns
        )

        config_kwargs: dict = {
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        generation_config = types.GenerateContentConfig(**config_kwargs)

        try:
            response_stream = await client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generation_config,
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
                    filter_err = _extract_finish_reason_error(response)
                    if filter_err is not None:
                        raise filter_err
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
        ``types.Part.from_function_response``, and generation resumes so the model
        can produce the final spoken reply. Returns the final text (never a
        raw function-call JSON blob).
        """
        try:
            client = _ensure_vertex_client(_location_for_model(model_name))
        except Exception as exc:
            raise VertexLlmError(f"Vertex init failed: {exc}", VertexLlmErrorType.UNKNOWN) from exc

        try:
            from google.genai import types
        except ImportError as exc:
            raise VertexLlmError(
                "google-genai is not installed. Add it to requirements.txt.",
                VertexLlmErrorType.UNKNOWN,
            ) from exc

        config_kwargs: dict = {
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
            "tools": [_build_calendly_tool()],
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        generation_config = types.GenerateContentConfig(**config_kwargs)

        max_turns = getattr(settings, "VOICE_LLM_HISTORY_MAX_TURNS", 20)
        contents = build_vertex_contents(conversation_history, prompt, None, max_turns=max_turns)

        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
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
                # google.genai's own automatic-function-calling loop builds this turn
                # with role="user" (not "function" — that was the old vertexai SDK's
                # convention). See google.genai.models.Models._generate_content_afc.
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=fc_name, response=tool_result)],
                )
            )

            follow_up = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
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
            client = _ensure_vertex_client(_location_for_model(model_name))
        except Exception as exc:
            raise VertexLlmError(f"Vertex init failed: {exc}", VertexLlmErrorType.UNKNOWN) from exc

        try:
            from google.genai import types
        except ImportError as exc:
            raise VertexLlmError(
                "google-genai is not installed.",
                VertexLlmErrorType.UNKNOWN,
            ) from exc

        config_kwargs: dict = {
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        generation_config = types.GenerateContentConfig(**config_kwargs)

        max_turns = getattr(settings, "VOICE_LLM_HISTORY_MAX_TURNS", 20)
        contents = build_vertex_contents(
            conversation_history, prompt, kb_context, max_turns=max_turns
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
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
