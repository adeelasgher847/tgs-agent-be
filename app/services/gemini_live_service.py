"""
Gemini Live (speech-to-speech native-audio) session wrapper.

Mirrors ``app/services/vertex_gemini_service.py``'s module shape but wraps a
*stateful* Live session instead of one-shot request/response calls. This
module has ZERO knowledge of Twilio/LiveKit/MULAW/telephony — it only speaks
raw PCM16 bytes in (16kHz, caller -> Gemini) and out (24kHz, Gemini ->
caller), exactly like ``vertex_gemini_service.py`` has zero knowledge of any
TTS/STT provider. Byte-level format conversion to/from telephony formats
lives in ``app/voice/gemini_live_audio_bridge.py``; wiring this session into
an actual call transport is out of scope for this module (see
``app/core/llm_models.py``'s ``GEMINI_LIVE_MODELS``/
``is_gemini_live_native_audio_model`` for the routing predicate used by
transport handlers).

Auth is per-model (see ``GEMINI_LIVE_MODELS``): two models use Vertex AI/ADC
(reusing ``vertex_gemini_service._ensure_vertex_client`` verbatim — same
client type, same location-keyed cache, no duplication), one uses the
Gemini Developer API with ``settings.GEMINI_API_KEY`` as a deliberate, scoped
exception (see that constant's docstring for why).
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any, Awaitable, Callable, Sequence

from app.core.config import settings
from app.core.llm_models import GEMINI_LIVE_MODELS
from app.core.logger import logger
from app.services.vertex_gemini_service import (
    VertexLlmError,
    VertexLlmErrorType,
    _classify_vertex_error,
    _ensure_vertex_client,
)

DEFAULT_VOICE_NAME = "Puck"

# All 30 Gemini Live prebuilt native-audio voices, confirmed against Google's
# published voice list (Zephyr, Kore, Orus, ... Sulafat). PrebuiltVoiceConfig
# rejects anything outside this set, so validate here rather than letting a
# bad agent.tts_voice_external_id value 400 the whole session at start().
VALID_VOICE_NAMES: frozenset[str] = frozenset(
    {
        "Zephyr", "Kore", "Orus", "Autonoe", "Umbriel", "Erinome", "Laomedeia",
        "Schedar", "Achird", "Sadachbia", "Puck", "Fenrir", "Aoede", "Enceladus",
        "Algieba", "Algenib", "Achernar", "Gacrux", "Zubenelgenubi", "Sadaltager",
        "Charon", "Leda", "Callirrhoe", "Iapetus", "Despina", "Rasalgethi",
        "Alnilam", "Pulcherrima", "Vindemiatrix", "Sulafat",
    }
)


def resolve_voice_name(requested: str | None) -> str:
    """
    Map an agent's configured voice to a valid Gemini Live prebuilt voice
    name, falling back to DEFAULT_VOICE_NAME when unset or not a real Gemini
    Live voice (e.g. an ElevenLabs/Rime/Google TTS voice ID left over from
    before the agent was switched to a native-audio model — those IDs mean
    nothing here and must never be passed through raw).
    """
    if not requested:
        return DEFAULT_VOICE_NAME
    for valid in VALID_VOICE_NAMES:
        if valid.lower() == requested.strip().lower():
            return valid
    # Distinguish "no voice configured" (silent, above) from "configured
    # voice doesn't resolve" (logged) — the latter usually means a leftover
    # ElevenLabs/Rime/Google TTS voice ID from before the agent was switched
    # to a native-audio model, but could also be a real typo/misconfiguration
    # worth a support/eng follow-up, so it shouldn't be indistinguishable
    # from the benign case in production logs.
    logger.info(
        "[GeminiLive] agent voice '%s' is not a valid Gemini Live voice name — "
        "falling back to %s",
        requested, DEFAULT_VOICE_NAME,
    )
    return DEFAULT_VOICE_NAME

_api_key_client_lock = threading.Lock()
_api_key_client: Any = None


def _ensure_api_key_client():
    """
    Lazily build (once, process-wide) and return a shared google.genai
    Client configured for the Gemini Developer API (API-key auth) — used
    only by ``gemini-3.1-flash-live-preview`` (see ``GEMINI_LIVE_MODELS``).
    Unlike ``_ensure_vertex_client``, this is NOT location-keyed: the
    Developer API is not project/location-scoped.
    """
    global _api_key_client
    if _api_key_client is not None:
        return _api_key_client
    with _api_key_client_lock:
        if _api_key_client is not None:
            return _api_key_client
        from google import genai

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for Gemini Live Developer-API models "
                "(e.g. gemini-3.1-flash-live-preview)."
            )
        _api_key_client = genai.Client(api_key=api_key)
        return _api_key_client


class GeminiLiveSession:
    """
    Transport-agnostic wrapper around one ``google.genai`` Live API session.

    Usage (by a future transport-handler wiring, not part of this task):
      session = GeminiLiveSession(agent.llm_model)
      await session.start(system_instruction=..., on_audio_chunk=..., on_interrupted=...)
      await session.send_audio(pcm16_16k_bytes)   # per caller-audio chunk
      ...
      await session.close()
    """

    def __init__(self, model_name: str) -> None:
        config = GEMINI_LIVE_MODELS.get(model_name)
        if config is None:
            raise ValueError(
                f"'{model_name}' is not a Gemini Live native-audio model. "
                f"Valid models: {sorted(GEMINI_LIVE_MODELS)}"
            )
        self.model_name = model_name
        self._auth_mode: str = config["auth"]
        self._location: str | None = config["location"]

        self._client: Any = None
        self._connect_cm: Any = None
        self._session: Any = None
        self._receive_task: asyncio.Task | None = None
        self._closed = False

        # Registered by start(); consumed by _receive_loop.
        self.on_audio_chunk: Callable[[bytes], Awaitable[None]] | None = None
        self.on_interrupted: Callable[[], Awaitable[None]] | None = None
        self.on_input_transcript: Callable[[str, bool], Awaitable[None]] | None = None
        self.on_output_transcript: Callable[[str, bool], Awaitable[None]] | None = None
        self.on_tool_call: Callable[[Sequence[Any]], Awaitable[None]] | None = None
        self.on_error: Callable[[VertexLlmError], Awaitable[None]] | None = None

    def _build_client(self):
        if self._auth_mode == "vertex":
            return _ensure_vertex_client(self._location)
        if self._auth_mode == "api_key":
            return _ensure_api_key_client()
        raise VertexLlmError(
            f"Unknown Gemini Live auth mode '{self._auth_mode}' for model {self.model_name}",
            VertexLlmErrorType.UNKNOWN,
        )

    async def start(
        self,
        *,
        system_instruction: str,
        voice_name: str = DEFAULT_VOICE_NAME,
        language_code: str | None = None,
        on_audio_chunk: Callable[[bytes], Awaitable[None]],
        on_interrupted: Callable[[], Awaitable[None]],
        on_input_transcript: Callable[[str, bool], Awaitable[None]] | None = None,
        on_output_transcript: Callable[[str, bool], Awaitable[None]] | None = None,
        on_tool_call: Callable[[Sequence[Any]], Awaitable[None]] | None = None,
        on_error: Callable[[VertexLlmError], Awaitable[None]] | None = None,
        tools: list | None = None,
    ) -> None:
        """
        Open the Live session: build the right client for this model's auth
        mode, connect, and start the background receive loop. Raises
        ``VertexLlmError`` on any setup failure (bad model, missing
        credentials, SDK/connect errors).
        """
        self.on_audio_chunk = on_audio_chunk
        self.on_interrupted = on_interrupted
        self.on_input_transcript = on_input_transcript
        self.on_output_transcript = on_output_transcript
        self.on_tool_call = on_tool_call
        self.on_error = on_error

        try:
            from google.genai import types
        except ImportError as exc:
            raise VertexLlmError(
                "google-genai is not installed. Add it to requirements.txt.",
                VertexLlmErrorType.UNKNOWN,
            ) from exc

        try:
            self._client = self._build_client()
        except VertexLlmError:
            raise
        except Exception as exc:
            raise VertexLlmError(
                f"Gemini Live client init failed: {exc}", VertexLlmErrorType.UNKNOWN
            ) from exc

        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            ),
            language_code=language_code,
        )
        config_kwargs: dict = {
            "response_modalities": ["AUDIO"],
            "speech_config": speech_config,
            "input_audio_transcription": types.AudioTranscriptionConfig(),
            "output_audio_transcription": types.AudioTranscriptionConfig(),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = tools
        live_config = types.LiveConnectConfig(**config_kwargs)

        try:
            self._connect_cm = self._client.aio.live.connect(
                model=self.model_name, config=live_config
            )
            self._session = await self._connect_cm.__aenter__()
        except Exception as exc:
            self._connect_cm = None
            raise _classify_vertex_error(exc) from exc

        # DIAGNOSTIC (temporary — root-cause investigation, strip before merge)
        logger.info(
            "[GeminiLiveDiag] __aenter__ returned session=%s model=%s — creating receive task",
            type(self._session).__name__, self.model_name,
        )
        self._receive_task = asyncio.create_task(self._receive_loop())
        # DIAGNOSTIC: confirm the task object is a real, live asyncio.Task
        # (not silently GC'd / not already done) right after creation, and
        # attach a done_callback so we get a log line the instant it exits
        # for ANY reason (normal StopAsyncIteration, exception, or cancel) —
        # today _receive_loop() can return normally (async-for simply runs
        # out of items) with zero log trace, which is indistinguishable from
        # "still running" without this.
        def _diag_receive_task_done(t: "asyncio.Task") -> None:
            try:
                exc = t.exception() if not t.cancelled() else None
            except Exception as e:  # exception() itself can raise
                exc = e
            logger.info(
                "[GeminiLiveDiag] receive_task DONE model=%s cancelled=%s exception=%r",
                self.model_name, t.cancelled(), exc,
            )
        self._receive_task.add_done_callback(_diag_receive_task_done)
        logger.info(
            "[GeminiLiveDiag] receive_task created model=%s task_done=%s",
            self.model_name, self._receive_task.done(),
        )

    async def send_audio(self, pcm16_16k_bytes: bytes) -> None:
        """Send one chunk of raw PCM16/16kHz/mono caller audio. Caller must
        have already converted to this format (e.g. via
        ``gemini_live_audio_bridge.mulaw8k_to_pcm16_16k``) — this module has
        zero telephony-format knowledge."""
        from google.genai import types

        # DIAGNOSTIC (temporary — root-cause investigation, strip before merge):
        # cheap counter to confirm the send side keeps succeeding for the
        # whole call (vs. silently hanging/failing) without logging every
        # single 20ms chunk.
        self._diag_send_count = getattr(self, "_diag_send_count", 0) + 1
        if self._diag_send_count % 50 == 1:  # ~ once/second at 20ms frames
            logger.info(
                "[GeminiLiveDiag] send_audio count=%d model=%s receive_task_done=%s",
                self._diag_send_count, self.model_name,
                self._receive_task.done() if self._receive_task else None,
            )

        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm16_16k_bytes, mime_type="audio/pcm;rate=16000")
        )

    async def send_text(self, text: str, turn_complete: bool = True) -> None:
        from google.genai import types

        content = types.Content(role="user", parts=[types.Part.from_text(text=text)])
        await self._session.send_client_content(turns=content, turn_complete=turn_complete)

    async def send_tool_response(self, function_responses) -> None:
        await self._session.send_tool_response(function_responses=function_responses)

    async def close(self) -> None:
        """
        Idempotent teardown — safe to call multiple times, and safe to call
        from a ``finally``/shutdown path. Never raises: all teardown errors
        are swallowed and logged.
        """
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
                logger.debug("[GeminiLiveSession] close: session teardown error (ignored): %s", exc)
            self._connect_cm = None

        self._session = None

    async def _receive_loop(self) -> None:
        """Consume ``session.receive()`` and demux each ``LiveServerMessage``
        into the registered callbacks. Any exception (SDK or otherwise) is
        translated to ``VertexLlmError`` and routed to ``on_error`` if set,
        else logged."""
        # DIAGNOSTIC (temporary — root-cause investigation, strip before merge)
        logger.info("[GeminiLiveDiag] _receive_loop ENTERED model=%s", self.model_name)
        msg_count = 0
        turn_count = 0
        try:
            # session.receive() is an async-generator that, per google-genai's
            # live.py implementation, yields only until the first
            # server_content.turn_complete=True message and then returns
            # (StopAsyncIteration) — it covers a single model turn, not the
            # whole call. It must be re-invoked for every subsequent turn,
            # hence the outer while loop below.
            while True:
                turn_count += 1
                turn_msg_count = 0
                async for message in self._session.receive():
                    msg_count += 1
                    turn_msg_count += 1
                    logger.info(
                        "[GeminiLiveDiag] server message #%d (turn %d) model=%s type=%s",
                        msg_count, turn_count, self.model_name, type(message).__name__,
                    )
                    try:
                        await self._handle_message(message)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "[GeminiLiveDiag] _handle_message raised on message #%d: %r",
                            msg_count, exc,
                        )
                        await self._report_error(exc)
                if turn_msg_count == 0:
                    # receive() returned with zero messages instead of raising —
                    # treat as a dead/closed connection rather than busy-loop
                    # re-invoking it forever.
                    logger.warning(
                        "[GeminiLiveDiag] receive() yielded no messages on turn %d "
                        "model=%s — ending receive loop (connection likely closed)",
                        turn_count, self.model_name,
                    )
                    break
        except asyncio.CancelledError:
            logger.info(
                "[GeminiLiveDiag] _receive_loop CANCELLED model=%s after %d messages/%d turns",
                self.model_name, msg_count, turn_count,
            )
            raise
        except Exception as exc:
            logger.info(
                "[GeminiLiveDiag] _receive_loop raised model=%s after %d messages/%d turns: %r",
                self.model_name, msg_count, turn_count, exc,
            )
            await self._report_error(exc)

    async def _report_error(self, exc: Exception) -> None:
        err = exc if isinstance(exc, VertexLlmError) else _classify_vertex_error(exc)
        if self.on_error is not None:
            await self.on_error(err)
        else:
            logger.error("[GeminiLiveSession] %s (model=%s): %s", err.error_type, self.model_name, err)

    async def _handle_message(self, message: Any) -> None:
        # DIAGNOSTIC (temporary — root-cause investigation, strip before merge):
        # log which top-level fields are present on this message — type/event
        # name only, never raw audio bytes or transcript text — so we can see
        # exactly what Gemini is sending without guessing at SDK field names.
        logger.info(
            "[GeminiLiveDiag] _handle_message model=%s setup_complete=%s "
            "has_server_content=%s has_tool_call=%s go_away=%s",
            self.model_name,
            bool(getattr(message, "setup_complete", None)),
            getattr(message, "server_content", None) is not None,
            getattr(message, "tool_call", None) is not None,
            bool(getattr(message, "go_away", None)),
        )

        if getattr(message, "setup_complete", None):
            logger.info("[GeminiLiveSession] setup_complete model=%s", self.model_name)

        server_content = getattr(message, "server_content", None)
        if server_content is not None:
            model_turn = getattr(server_content, "model_turn", None)
            # DIAGNOSTIC
            logger.info(
                "[GeminiLiveDiag] server_content model=%s has_model_turn=%s "
                "interrupted=%s turn_complete=%s",
                self.model_name,
                model_turn is not None,
                bool(getattr(server_content, "interrupted", False)),
                bool(getattr(server_content, "turn_complete", False)),
            )
            if model_turn is not None:
                parts = getattr(model_turn, "parts", None) or []
                for part in parts:
                    inline_data = getattr(part, "inline_data", None)
                    data = getattr(inline_data, "data", None) if inline_data is not None else None
                    # DIAGNOSTIC — byte count only, never raw audio.
                    logger.info(
                        "[GeminiLiveDiag] model_turn part model=%s has_inline_data=%s "
                        "data_len=%s on_audio_chunk_registered=%s",
                        self.model_name,
                        inline_data is not None,
                        len(data) if data else 0,
                        self.on_audio_chunk is not None,
                    )
                    if data and self.on_audio_chunk is not None:
                        await self.on_audio_chunk(data)

            if getattr(server_content, "interrupted", False) and self.on_interrupted is not None:
                await self.on_interrupted()

            input_transcript = getattr(server_content, "input_transcription", None)
            if input_transcript is not None and self.on_input_transcript is not None:
                await self.on_input_transcript(
                    getattr(input_transcript, "text", "") or "",
                    bool(getattr(input_transcript, "finished", False)),
                )

            output_transcript = getattr(server_content, "output_transcription", None)
            if output_transcript is not None and self.on_output_transcript is not None:
                await self.on_output_transcript(
                    getattr(output_transcript, "text", "") or "",
                    bool(getattr(output_transcript, "finished", False)),
                )

        tool_call = getattr(message, "tool_call", None)
        if tool_call is not None and self.on_tool_call is not None:
            function_calls = getattr(tool_call, "function_calls", None) or []
            await self.on_tool_call(function_calls)

        if getattr(message, "go_away", None):
            logger.warning(
                "[GeminiLiveSession] go_away received model=%s — no action taken (phase 1)",
                self.model_name,
            )
