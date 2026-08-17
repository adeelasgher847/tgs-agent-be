"""
OpenAI Realtime (speech-to-speech native-audio) session wrapper.

Mirrors ``app/services/gemini_live_service.py``'s module shape/callback
contract (``GeminiLiveSession``) so ``VoiceOrchestrator`` can wire either
provider into the same transport handlers with parallel, not shared, code
paths (per this repo's convention of keeping the two Twilio/LiveKit
transport handlers "deliberately parallel, not shared via inheritance" —
same idea applied one layer down, at the provider level).

This module has ZERO knowledge of Twilio/LiveKit/telephony framing — it
only speaks whatever raw audio bytes the caller tells it to use via
``audio_format`` (``"audio/pcmu"`` — mu-law 8kHz, a direct match for
Twilio's own wire format, so the Twilio transport needs NO resampling
bridge at all; or ``"audio/pcm"`` — PCM16 24kHz, symmetric in both
directions, which is what the LiveKit browser transport's publisher speaks
for Gemini Live too). See ``app/core/llm_models.py``'s
``OPENAI_REALTIME_MODELS``/``is_openai_realtime_model`` for the routing
predicate used by transport handlers.

Deliberate divergences from ``GeminiLiveSession`` (do not "fix" these to
match Gemini — they are correct as written; see each divergence's own
docstring for the evidence):

  * ``_receive_loop`` is a single flat ``while True: recv(); handle()``
    loop — ``AsyncRealtimeConnection.recv()`` returns exactly one parsed
    server event per call with no per-turn generator-exhaustion quirk,
    unlike ``google.genai``'s ``session.receive()`` (whose per-turn
    re-invocation requirement was the root cause of a real "silent after
    turn 1" bug this repo shipped and then fixed for Gemini Live — see
    that module's ``_receive_loop`` docstring for the incident). Wrapping
    this loop in Gemini's outer per-turn ``while True`` here would be
    solving a problem this SDK doesn't have.
  * Transcripts are captured directly off
    ``conversation.item.input_audio_transcription.completed`` /
    ``response.output_audio_transcript.done`` — each fires exactly once
    per utterance with the FULL final text in ``.transcript``. No
    fragment-buffering/reassembly workaround is needed (unlike Gemini
    Live's per-fragment ``finished``-flag unreliability) — do not port that
    buffering pattern here, it would be solving a problem this API
    contract doesn't have either.
  * Barge-in is largely server-driven: session config sets
    ``turn_detection.interrupt_response=True``, so OpenAI cancels its own
    in-flight response server-side the moment it detects caller speech.
    ``on_interrupted`` (wired to ``input_audio_buffer.speech_started``) is
    still surfaced so the transport can locally clear whatever audio it
    has already buffered/sent (mirrors Gemini Live's ``on_interrupted``
    contract at the VoiceOrchestrator layer, even though the server-side
    half of the interruption is automatic here and manual for Gemini).
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import enum
import json
from typing import Any, Awaitable, Callable

from app.core.llm_models import OPENAI_REALTIME_MODELS
from app.core.logger import logger
from app.core.openai_client import get_async_openai_client

DEFAULT_VOICE_NAME = "marin"

# Confirmed via the installed openai==2.36.0 SDK
# (openai.types.realtime.realtime_audio_output.RealtimeAudioConfigOutput's
# `voice` field / OpenAI's Realtime voices list). OpenAI recommends `marin`/
# `cedar` for best quality on the newer Realtime models — used as the
# default here rather than a legacy voice like `alloy`.
VALID_VOICE_NAMES: frozenset[str] = frozenset(
    {
        "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer",
        "verse", "marin", "cedar",
    }
)

# gpt-realtime-2 is documented (SDK-side, via RealtimeReasoning/
# reasoning_effort) as "reasoning-capable" — an uncapped reasoning budget on
# a live voice turn risks noticeable dead air before the model starts
# speaking, same rationale as app.services.openai_service.py's
# _REASONING_EFFORT_BY_MODEL cap for the gpt-5.x Chat Completions family
# (see that module's docstring for the full incident writeup this pattern
# is modeled on). `gpt-realtime` (the original, non-"-2" model) has no
# reasoning config at all, so it is deliberately absent from this map —
# sending a `reasoning` block to a non-reasoning-capable model is not
# something this integration has verified is safe, so it is simply never
# sent for that model.
_REASONING_EFFORT_BY_MODEL: dict[str, str] = {
    "gpt-realtime-2": "low",
}

# Twilio's own wire format (mu-law 8kHz) — passed straight through with NO
# conversion, unlike Gemini Live's forced PCM16 resample round-trip (see
# app/voice/gemini_live_audio_bridge.py). Confirmed via
# openai.types.realtime.realtime_audio_formats.AudioPCMU: `audio/pcmu` is a
# natively supported input AND output format.
TWILIO_AUDIO_FORMAT = "audio/pcmu"
# LiveKit browser transport's publisher format — PCM16, 24kHz only
# (confirmed via AudioPCM: `rate` is `Literal[24000]`), symmetric for both
# input and output (unlike Gemini Live's asymmetric 16kHz-in/24kHz-out).
LIVEKIT_AUDIO_FORMAT = "audio/pcm"
LIVEKIT_AUDIO_RATE_HZ = 24000


def resolve_voice_name(requested: str | None) -> str:
    """
    Map an agent's configured voice to a valid OpenAI Realtime voice name,
    falling back to DEFAULT_VOICE_NAME when unset or not a real Realtime
    voice (e.g. an ElevenLabs/Rime/Google TTS voice ID left over from before
    the agent was switched to a native-audio model). Mirrors
    gemini_live_service.resolve_voice_name's fallback-and-log-on-mismatch
    pattern exactly.
    """
    if not requested:
        return DEFAULT_VOICE_NAME
    for valid in VALID_VOICE_NAMES:
        if valid.lower() == requested.strip().lower():
            return valid
    logger.info(
        "[OpenAIRealtime] agent voice '%s' is not a valid OpenAI Realtime voice name — "
        "falling back to %s",
        requested, DEFAULT_VOICE_NAME,
    )
    return DEFAULT_VOICE_NAME


class OpenAIRealtimeErrorType(str, enum.Enum):
    QUOTA = "quota"
    TIMEOUT = "timeout"
    AUTH = "auth"
    CONNECTION_CLOSED = "connection_closed"
    UNKNOWN = "unknown"


class OpenAIRealtimeError(Exception):
    def __init__(
        self, message: str, error_type: OpenAIRealtimeErrorType = OpenAIRealtimeErrorType.UNKNOWN
    ) -> None:
        super().__init__(message)
        self.error_type = error_type


def _classify_openai_realtime_error(exc: Exception) -> OpenAIRealtimeError:
    """Mirrors gemini_live_service._classify_vertex_error's shape/intent,
    against the openai-python SDK's own exception hierarchy instead of
    google.genai's."""
    import openai as openai_sdk

    msg = str(exc)
    if isinstance(exc, openai_sdk.RateLimitError):
        return OpenAIRealtimeError(f"OpenAI Realtime quota exceeded: {msg}", OpenAIRealtimeErrorType.QUOTA)
    if isinstance(exc, openai_sdk.APITimeoutError):
        return OpenAIRealtimeError(f"OpenAI Realtime timeout: {msg}", OpenAIRealtimeErrorType.TIMEOUT)
    if isinstance(exc, openai_sdk.AuthenticationError):
        return OpenAIRealtimeError(f"OpenAI Realtime auth failed: {msg}", OpenAIRealtimeErrorType.AUTH)
    if isinstance(exc, getattr(openai_sdk, "WebSocketConnectionClosedError", ())):
        return OpenAIRealtimeError(f"OpenAI Realtime connection closed: {msg}", OpenAIRealtimeErrorType.CONNECTION_CLOSED)

    name = type(exc).__name__
    lowered = msg.lower()
    if "quota" in lowered or ("rate" in lowered and "limit" in lowered):
        return OpenAIRealtimeError(f"OpenAI Realtime quota: {msg}", OpenAIRealtimeErrorType.QUOTA)
    if "timeout" in lowered or "timed out" in lowered:
        return OpenAIRealtimeError(f"OpenAI Realtime timeout: {msg}", OpenAIRealtimeErrorType.TIMEOUT)
    if "closed" in lowered or "ConnectionClosed" in name:
        return OpenAIRealtimeError(f"OpenAI Realtime connection closed: {msg}", OpenAIRealtimeErrorType.CONNECTION_CLOSED)
    return OpenAIRealtimeError(f"OpenAI Realtime error: {msg}", OpenAIRealtimeErrorType.UNKNOWN)


def _build_calendly_tool_params() -> list[dict]:
    """
    OpenAI Realtime function-tool declarations for the Calendly booking
    flow — same two tools/parameter shapes as
    ``vertex_gemini_service._build_calendly_tool()``, translated to the
    Realtime API's flat ``{type: "function", name, description,
    parameters}`` tool shape (confirmed via
    ``openai.types.realtime.RealtimeFunctionToolParam``) instead of
    google.genai's ``FunctionDeclaration`` wrapper.
    """
    return [
        {
            "type": "function",
            "name": "check_availability",
            "description": (
                "Check available appointment slots on the connected Calendly calendar. "
                "Slot length is fixed by the connected Calendly event type — it cannot "
                "be requested per-call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to check availability for, as YYYY-MM-DD (or 'today'/'tomorrow').",
                    },
                },
                "required": ["date"],
            },
        },
        {
            "type": "function",
            "name": "book_appointment",
            "description": "Schedule an appointment on Calendly for a previously offered slot.",
            "parameters": {
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
        },
    ]


def _audio_format_dict(audio_format: str) -> dict:
    if audio_format == LIVEKIT_AUDIO_FORMAT:
        return {"type": LIVEKIT_AUDIO_FORMAT, "rate": LIVEKIT_AUDIO_RATE_HZ}
    return {"type": audio_format}


class OpenAIRealtimeSession:
    """
    Transport-agnostic wrapper around one OpenAI Realtime API connection.

    Usage (mirrors GeminiLiveSession):
      session = OpenAIRealtimeSession(agent.llm_model)
      await session.start(system_instruction=..., audio_format="audio/pcmu",
                           on_audio_chunk=..., on_interrupted=...)
      await session.send_audio(mulaw_or_pcm16_bytes)   # per caller-audio chunk
      ...
      await session.close()
    """

    def __init__(self, model_name: str) -> None:
        if model_name not in OPENAI_REALTIME_MODELS:
            raise ValueError(
                f"'{model_name}' is not an OpenAI Realtime speech-to-speech model. "
                f"Valid models: {sorted(OPENAI_REALTIME_MODELS)}"
            )
        self.model_name = model_name

        self._client: Any = None
        self._connect_cm: Any = None
        self._connection: Any = None
        self._receive_task: asyncio.Task | None = None
        self._closed = False

        # Registered by start(); consumed by _receive_loop.
        self.on_audio_chunk: Callable[[bytes], Awaitable[None]] | None = None
        self.on_interrupted: Callable[[], Awaitable[None]] | None = None
        # Demuxed from OpenAI's own input_audio_buffer.speech_stopped event
        # (see _handle_message). Not currently wired by VoiceOrchestrator —
        # nothing there needs a "caller stopped talking" signal yet — but the
        # session-level plumbing is here and tested so a future turn-boundary
        # or analytics consumer doesn't have to add wire-protocol handling.
        self.on_speech_stopped: Callable[[], Awaitable[None]] | None = None
        # Fires once per utterance with the FULL final transcript text — no
        # `finished`-flag/fragment-buffering dance (see module docstring).
        self.on_input_transcript: Callable[[str], Awaitable[None]] | None = None
        self.on_output_transcript: Callable[[str], Awaitable[None]] | None = None
        self.on_tool_call: Callable[[str, str, str], Awaitable[None]] | None = None
        self.on_error: Callable[[OpenAIRealtimeError], Awaitable[None]] | None = None

    def _build_client(self):
        return get_async_openai_client()

    async def start(
        self,
        *,
        system_instruction: str,
        audio_format: str = LIVEKIT_AUDIO_FORMAT,
        voice_name: str = DEFAULT_VOICE_NAME,
        on_audio_chunk: Callable[[bytes], Awaitable[None]],
        on_interrupted: Callable[[], Awaitable[None]],
        on_speech_stopped: Callable[[], Awaitable[None]] | None = None,
        on_input_transcript: Callable[[str], Awaitable[None]] | None = None,
        on_output_transcript: Callable[[str], Awaitable[None]] | None = None,
        on_tool_call: Callable[[str, str, str], Awaitable[None]] | None = None,
        on_error: Callable[[OpenAIRealtimeError], Awaitable[None]] | None = None,
        tools: list[dict] | None = None,
    ) -> None:
        """
        Open the Realtime connection, send the initial `session.update`, and
        start the background receive loop. Raises ``OpenAIRealtimeError`` on
        any setup failure (bad model, missing credentials, SDK/connect
        errors) — mirrors GeminiLiveSession.start()'s contract exactly so
        VoiceOrchestrator's pre-session-failure fallback handling can stay
        provider-agnostic at the call site.
        """
        self.on_audio_chunk = on_audio_chunk
        self.on_interrupted = on_interrupted
        self.on_speech_stopped = on_speech_stopped
        self.on_input_transcript = on_input_transcript
        self.on_output_transcript = on_output_transcript
        self.on_tool_call = on_tool_call
        self.on_error = on_error

        try:
            self._client = self._build_client()
        except Exception as exc:
            raise OpenAIRealtimeError(
                f"OpenAI Realtime client init failed: {exc}", OpenAIRealtimeErrorType.UNKNOWN
            ) from exc

        try:
            self._connect_cm = self._client.realtime.connect(model=self.model_name)
            self._connection = await self._connect_cm.__aenter__()
        except Exception as exc:
            self._connect_cm = None
            raise _classify_openai_realtime_error(exc) from exc

        session_config: dict[str, Any] = {
            "type": "realtime",
            "model": self.model_name,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": _audio_format_dict(audio_format),
                    "transcription": {"model": "gpt-4o-mini-transcribe"},
                    "turn_detection": {
                        "type": "server_vad",
                        "interrupt_response": True,
                        "create_response": True,
                    },
                },
                "output": {
                    "format": _audio_format_dict(audio_format),
                    "voice": voice_name,
                },
            },
        }
        if system_instruction:
            session_config["instructions"] = system_instruction
        if tools:
            session_config["tools"] = tools
        effort = _REASONING_EFFORT_BY_MODEL.get(self.model_name)
        if effort:
            session_config["reasoning"] = {"effort": effort}

        try:
            await self._connection.send({"type": "session.update", "session": session_config})
        except Exception as exc:
            raise _classify_openai_realtime_error(exc) from exc

        self._receive_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send one chunk of raw caller audio, already encoded in whatever
        format this session was started with (`audio_format` — mu-law 8kHz
        for Twilio, PCM16 24kHz for LiveKit). This module has zero
        telephony-format knowledge beyond that string."""
        if not audio_bytes:
            return
        await self._connection.send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_bytes).decode("ascii"),
        })

    async def send_text(self, text: str, respond: bool = True) -> None:
        """
        Inject a text turn into the conversation via
        `conversation.item.create` (role="user"). Unlike GeminiLiveSession's
        single `send_client_content(turn_complete=...)` call, the Realtime
        API separates "add this item to the conversation" from "generate a
        response" — a context-only injection (e.g. mid-call RAG refresh)
        must send ONLY the item-create event, never a following
        `response.create`, or OpenAI will speak the raw injected text as if
        it were a real reply. `respond=True` (the default — used for e.g.
        the scripted greeting, which SHOULD be spoken) sends both.
        """
        if not text:
            return
        await self._connection.send({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        })
        if respond:
            await self._connection.send({"type": "response.create"})

    async def send_tool_response(self, call_id: str, output: dict) -> None:
        """
        Resolve a function call: `conversation.item.create` with a
        `function_call_output` item, then an explicit `response.create` to
        prompt the model to continue — unlike GeminiLiveSession's single
        `send_tool_response(function_responses=...)` call, the Realtime API
        requires this second event to actually resume generation.
        """
        await self._connection.send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output),
            },
        })
        await self._connection.send({"type": "response.create"})

    async def close(self) -> None:
        """Idempotent teardown — safe to call multiple times, and safe to
        call from a finally/shutdown path. Never raises: all teardown
        errors are swallowed and logged. Mirrors GeminiLiveSession.close()."""
        if self._closed:
            return
        self._closed = True

        if self._receive_task is not None:
            self._receive_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._receive_task
            self._receive_task = None

        if self._connect_cm is not None:
            try:
                await self._connect_cm.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("[OpenAIRealtimeSession] close: teardown error (ignored): %s", exc)
            self._connect_cm = None

        self._connection = None

    async def _receive_loop(self) -> None:
        """
        Flat receive loop — see module docstring for why this is NOT
        shaped like GeminiLiveSession's per-turn outer loop.
        `connection.recv()` raising (connection genuinely dropped, or any
        SDK-level error) ends the loop after routing to on_error/logging —
        there is no "zero messages, retry" case to guard against here since
        each recv() call either returns exactly one event or raises.
        """
        try:
            while True:
                try:
                    event = await self._connection.recv()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._report_error(exc)
                    return
                try:
                    await self._handle_event(event)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._report_error(exc)
        except asyncio.CancelledError:
            raise

    async def _report_error(self, exc: Exception) -> None:
        err = exc if isinstance(exc, OpenAIRealtimeError) else _classify_openai_realtime_error(exc)
        if self.on_error is not None:
            await self.on_error(err)
        else:
            logger.error("[OpenAIRealtimeSession] %s (model=%s): %s", err.error_type, self.model_name, err)

    async def _handle_event(self, event: Any) -> None:
        etype = getattr(event, "type", None)

        if etype == "response.output_audio.delta":
            delta = getattr(event, "delta", None)
            if delta and self.on_audio_chunk is not None:
                await self.on_audio_chunk(base64.b64decode(delta))
            return

        if etype == "input_audio_buffer.speech_started":
            if self.on_interrupted is not None:
                await self.on_interrupted()
            return

        if etype == "input_audio_buffer.speech_stopped":
            if self.on_speech_stopped is not None:
                await self.on_speech_stopped()
            return

        if etype == "conversation.item.input_audio_transcription.completed":
            transcript = getattr(event, "transcript", "") or ""
            if transcript and self.on_input_transcript is not None:
                await self.on_input_transcript(transcript)
            return

        if etype == "response.output_audio_transcript.done":
            transcript = getattr(event, "transcript", "") or ""
            if transcript and self.on_output_transcript is not None:
                await self.on_output_transcript(transcript)
            return

        if etype == "response.function_call_arguments.done":
            if self.on_tool_call is not None:
                call_id = getattr(event, "call_id", None) or ""
                name = getattr(event, "name", None) or ""
                arguments = getattr(event, "arguments", None) or "{}"
                await self.on_tool_call(call_id, name, arguments)
            return

        if etype == "error":
            error_obj = getattr(event, "error", None)
            message = getattr(error_obj, "message", None) or str(error_obj) or "unknown Realtime API error"
            await self._report_error(OpenAIRealtimeError(message, OpenAIRealtimeErrorType.UNKNOWN))
            return

        # All other event types (session.created, response.created,
        # conversation.item.created, rate_limits.updated, delta variants we
        # deliberately don't consume, etc.) are expected, benign chatter —
        # no action needed.
