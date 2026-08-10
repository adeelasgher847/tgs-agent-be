"""
ElevenLabs `stream-input` WebSocket session (Phase 6-3).

A persistent, incremental synthesis session for ONE agent turn — the
production counterpart to the Phase 6-2 spike
(scripts/spikes/elevenlabs_websocket_spike.py), reusing its empirically
validated protocol usage (init message shape, `auto_mode=true` query param,
flush-at-end, clean-close behavior) rather than reinventing it.

Ownership / lifecycle contract (enforced by app.voice.tts_pipeline.TtsPipeline
— see `_try_elevenlabs_ws_route`):
    - Opened lazily by the first TTS chunk of a turn that needs ElevenLabs
      synthesis. Never opened eagerly at turn start.
    - `send_text()` is called once per pipeline chunk — this class does NOT
      re-implement any sentence/time-flush chunking; it only relays text the
      caller has already decided to send.
    - `finalize()` is called exactly once, when the pipeline's true
      turn-final chunk is processed — sends the ElevenLabs flush signal.
    - `iter_audio()` yields raw audio bytes as they arrive and ends when the
      server reports `isFinal: true` (or the connection closes/errors).
    - `aclose()` tears the session down. A session MUST NOT be reused for a
      second turn — `send_text`/`finalize` raise `ElevenLabsWebSocketSessionError`
      once closed, matching the closed-socket-rejects-writes behavior the
      Phase 6-2 spike confirmed live.

Concurrency:
    - Single-writer guarantee: all outbound messages (init aside, which is
      sent synchronously during `start()` before any concurrent writer could
      exist) go through one internal queue drained by a single dedicated
      writer task — so multiple pipeline chunks racing to call `send_text()`
      concurrently can never interleave writes on the socket, and a slow
      write can never block the caller (e.g. the LLM-streaming coroutine)
      since `send_text()`/`finalize()` only enqueue and return immediately.
    - The receive loop runs as its own independent task, decoupled from the
      writer, so reads and writes never block one another.

What this class deliberately does NOT do:
    - It does not decide chunk boundaries (that's the pipeline's job, driven
      by VOICE_TTS_FLUSH_MIN_WORDS/MAX_WORDS/TIME_FLUSH_SEC — untouched here).
    - It does not attempt to correlate a specific audio byte range back to a
      specific input `send_text()` call. ElevenLabs' single-context
      `stream-input` protocol with `auto_mode=true` does not expose that
      correlation (the multi-context `CloseContextClient` variant that would
      is explicitly out of scope per Phase 6-1). The caller (TtsPipeline)
      accounts for this by having exactly one chunk ("the owner") drain
      `iter_audio()` for the whole turn — see `_try_elevenlabs_ws_route`'s
      docstring for the concrete routing/ownership design.
    - It does not apply humanization pacing/pause frames — those remain
      entirely the pipeline/transport layer's responsibility.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, AsyncIterator, Dict

from app.core.logger import logger
from app.services.elevenlabs_service import elevenlabs_service

_WS_BASE = "wss://api.elevenlabs.io/v1/text-to-speech"

# Same connect-timeout budget as the rest of the hot TTS path (elevenlabs_service
# uses request_timeout_seconds=25 for HTTP; opening a WS handshake is normally
# sub-second, so a short timeout here fails fast into the HTTP fallback path
# instead of stalling a turn for a hung network call).
_WS_CONNECT_TIMEOUT_S: float = 5.0

# Mirrors the Phase 6-2 spike's validated query params exactly.
_AUTO_MODE = "true"
_INACTIVITY_TIMEOUT_S = 20

# Voice-settings keys the `stream-input` WebSocket protocol actually accepts
# in its init message (validated live by the Phase 6-2 spike's
# DEFAULT_VOICE_SETTINGS). Deliberately excludes HTTP-only fields such as
# `previous_text`/`next_text`/`previous_request_ids`/`next_request_ids` —
# those are REST-continuity-only per Phase 6-1's research and are not sent
# here; the WS session's own persistent connection is the continuity
# mechanism for WS-routed chunks instead.
_WS_VOICE_SETTINGS_KEYS = (
    "stability",
    "similarity_boost",
    "style",
    "use_speaker_boost",
    "speed",
)


class ElevenLabsWebSocketSessionError(RuntimeError):
    """Session failed to open/initialize, or was used after being closed."""


class ElevenLabsWebSocketSession:
    """One ElevenLabs `stream-input` WebSocket session, scoped to a single
    agent turn. See module docstring for the full lifecycle contract."""

    def __init__(
        self,
        *,
        voice_external_id: str,
        model_id: str | None = None,
        output_format: str | None = None,
        voice_settings: Dict[str, Any] | None = None,
        api_key_override: str | None = None,
    ) -> None:
        self._voice_external_id = voice_external_id
        self._model_id = model_id or "eleven_flash_v2_5"
        self._output_format = output_format or "ulaw_8000"
        self._voice_settings = {
            k: v
            for k, v in dict(voice_settings or {}).items()
            if k in _WS_VOICE_SETTINGS_KEYS
        }
        self._api_key_override = api_key_override

        self._ws: Any = None
        self._recv_task: asyncio.Task | None = None
        self._writer_task: asyncio.Task | None = None
        self._audio_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()
        self._write_queue: "asyncio.Queue[Dict[str, Any] | None]" = asyncio.Queue()

        self._started = False
        self._closed = False
        # Set when a write to the socket fails mid-session (transient
        # network blip, ConnectionClosed, protocol error, etc.) — DISTINCT
        # from `_closed` (which means "fully torn down"). See _writer_loop:
        # a write failure marks this immediately, unblocks any iter_audio()
        # consumer immediately, and schedules a full aclose() as a separate
        # task (the writer task can't await itself inside aclose()).
        self._write_failed = False
        self._final_flush_sent = False
        self._open_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_url(self) -> str:
        return (
            f"{_WS_BASE}/{self._voice_external_id}/stream-input"
            f"?model_id={self._model_id}&output_format={self._output_format}"
            f"&auto_mode={_AUTO_MODE}&inactivity_timeout={_INACTIVITY_TIMEOUT_S}"
        )

    async def start(self, initial_text: str | None = None) -> None:
        """
        Open the WebSocket, send the init message, and (if provided) relay
        the first chunk's text. Idempotent — a second call is a no-op once
        already started. Raises ElevenLabsWebSocketSessionError on any
        failure (auth resolution, connect, or init send) so the caller can
        fall back to the HTTP path for this turn.
        """
        if self._closed:
            raise ElevenLabsWebSocketSessionError(
                "cannot start an already-closed session"
            )
        if self._started:
            if initial_text:
                await self.send_text(initial_text)
            return

        async with self._open_lock:
            if self._started:
                if initial_text:
                    await self.send_text(initial_text)
                return
            try:
                import websockets  # local import: only needed on this (opt-in) path
            except (
                ImportError
            ) as exc:  # pragma: no cover - dependency is pinned in requirements.txt
                raise ElevenLabsWebSocketSessionError(
                    "the 'websockets' package is not installed"
                ) from exc

            try:
                api_key = elevenlabs_service.get_api_key(self._api_key_override)
            except Exception as exc:
                raise ElevenLabsWebSocketSessionError(
                    f"could not resolve API key: {exc}"
                ) from exc

            url = self._build_url()
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(url, additional_headers={"xi-api-key": api_key}),
                    timeout=_WS_CONNECT_TIMEOUT_S,
                )
                init_payload = {"text": " ", "voice_settings": self._voice_settings}
                await self._ws.send(json.dumps(init_payload))
            except Exception as exc:
                self._closed = True
                raise ElevenLabsWebSocketSessionError(
                    f"failed to open ElevenLabs WS streaming session: {exc}"
                ) from exc

            self._started = True
            self._recv_task = asyncio.create_task(
                self._receive_loop(), name="elevenlabs_ws_recv"
            )
            self._writer_task = asyncio.create_task(
                self._writer_loop(), name="elevenlabs_ws_writer"
            )

        if initial_text:
            await self.send_text(initial_text)

    async def send_text(self, text: str) -> None:
        """
        Relay one already-decided pipeline chunk's text into the session.
        Enqueues onto the single-writer queue and returns immediately — never
        blocks on the network, so a slow/stuck write can't stall whatever
        coroutine is producing chunks (e.g. LLM streaming).
        """
        if self._write_failed:
            raise ElevenLabsWebSocketSessionError(
                "cannot write to an ElevenLabs WS session after a write failure"
            )
        if self._closed:
            raise ElevenLabsWebSocketSessionError(
                "cannot write to a closed ElevenLabs WS session"
            )
        if not self._started:
            raise ElevenLabsWebSocketSessionError("session not started")
        if not text:
            return
        payload_text = text if text.endswith(" ") else f"{text} "
        await self._write_queue.put({"text": payload_text})

    async def finalize(self) -> None:
        """
        Send the ElevenLabs flush signal exactly once, at true turn-end.
        Idempotent — safe to call multiple times or defensively when the
        session is merely already closed/already flushed. NOT idempotent
        after a write failure — raises instead, so a caller relying on
        finalize() to actually flush audio finds out immediately rather than
        silently no-op'ing (a dropped flush would otherwise strand the WS
        owner's iter_audio() consumer waiting for `isFinal` that will never
        arrive from a dead write path).
        """
        if self._write_failed:
            raise ElevenLabsWebSocketSessionError(
                "cannot finalize an ElevenLabs WS session after a write failure"
            )
        if self._closed or self._final_flush_sent:
            return
        if not self._started:
            return
        self._final_flush_sent = True
        await self._write_queue.put({"text": "", "flush": True})

    def iter_audio(self) -> AsyncIterator[bytes]:
        """
        Async-iterate raw audio bytes as they arrive from the server. Ends
        (StopAsyncIteration) once the server reports `isFinal: true` or the
        connection closes/errors. Consuming this fully drains one turn's
        audio — do not call more than once per session.
        """
        return self._audio_generator()

    async def _audio_generator(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def aclose(self) -> None:
        """
        Tear the session down: stop the writer, close the socket, cancel the
        receive task, and unblock any pending `iter_audio()` consumer.
        Idempotent — safe to call more than once (e.g. once from the owning
        chunk's own cleanup and once defensively from a turn-reset path).
        """
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True

        try:
            self._write_queue.put_nowait(None)
        except Exception:  # noqa: S110 - best-effort shutdown signal
            pass
        if self._writer_task is not None and not self._writer_task.done():
            self._writer_task.cancel()
            try:
                await self._writer_task
            except (
                asyncio.CancelledError,
                Exception,
            ):  # noqa: S110 - best-effort teardown
                pass

        if self._ws is not None:
            try:
                await self._ws.close(code=1000)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - best-effort close, already tearing down
                logger.debug("[ElevenLabsWS] close() raised during teardown: %s", exc)

        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (
                asyncio.CancelledError,
                Exception,
            ):  # noqa: S110 - best-effort teardown
                pass

        # Unblock any consumer still awaiting audio_queue.get().
        try:
            self._audio_queue.put_nowait(None)
        except Exception:  # noqa: S110 - best-effort
            pass

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def write_failed(self) -> bool:
        """True if a write to the socket failed mid-session. Consulted by
        TtsPipeline after the WS owner's iter_audio() unblocks, to
        distinguish "turn completed normally (isFinal received)" from "the
        iterator was force-unblocked because the write path died" — so the
        latter surfaces as a real, visible error instead of looking like a
        normal, silent turn completion."""
        return self._write_failed

    # ------------------------------------------------------------------
    # Internal tasks
    # ------------------------------------------------------------------

    async def _writer_loop(self) -> None:
        """Single dedicated writer — the only task that ever calls
        `self._ws.send()`, guaranteeing no concurrent writers on one socket.

        On a write failure (transient network blip, ConnectionClosed,
        protocol error, etc.): mark the session failed immediately, push the
        terminal sentinel onto `_audio_queue` immediately so ANY current or
        future `iter_audio()` consumer unblocks right away (never left
        hanging on the server's own 20s inactivity_timeout as the only
        backstop — that backstop only helps if the receive side also fails,
        which isn't guaranteed for a write-only failure), and schedule a
        full `aclose()` on a SEPARATE task (this task cannot safely
        cancel-and-await itself from inside `aclose()`).
        """
        try:
            while True:
                msg = await self._write_queue.get()
                if msg is None:
                    break
                if self._ws is None:
                    break
                try:
                    await self._ws.send(json.dumps(msg))
                except Exception as exc:
                    logger.error(
                        "[ElevenLabsWS] write failed mid-session — marking "
                        "session failed and unblocking any iter_audio() "
                        "consumer immediately: %s",
                        exc,
                    )
                    self._write_failed = True
                    try:
                        self._audio_queue.put_nowait(None)
                    except Exception:  # noqa: S110 - best-effort unblock
                        pass
                    try:
                        asyncio.create_task(self.aclose())
                    except RuntimeError:  # noqa: S110 - no running loop (defensive)
                        pass
                    break
        except asyncio.CancelledError:
            pass

    async def _receive_loop(self) -> None:
        """Independent of the writer — decodes AudioOutput messages and
        pushes raw bytes onto the audio queue; detects `isFinal` to end the
        turn's audio cleanly."""
        try:
            async for raw_msg in self._ws:
                try:
                    msg = json.loads(raw_msg)
                except (json.JSONDecodeError, TypeError):
                    continue

                audio_b64 = msg.get("audio")
                is_final = bool(msg.get("isFinal"))

                if audio_b64:
                    try:
                        await self._audio_queue.put(base64.b64decode(audio_b64))
                    except (
                        Exception
                    ) as exc:  # noqa: BLE001 - malformed payload must not kill the loop
                        logger.warning(
                            "[ElevenLabsWS] failed to decode audio chunk: %s", exc
                        )

                if is_final:
                    break
        except asyncio.CancelledError:
            raise
        except (
            Exception
        ) as exc:  # noqa: BLE001 - connection error/close; surface as end-of-audio
            logger.debug("[ElevenLabsWS] receive loop ended: %s", exc)
        finally:
            # Terminal sentinel — always unblocks iter_audio(), whether we
            # exited via isFinal, an error, or cancellation.
            try:
                self._audio_queue.put_nowait(None)
            except Exception:  # noqa: S110 - best-effort
                pass
