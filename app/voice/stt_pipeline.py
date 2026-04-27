import asyncio
from typing import Awaitable, Callable, Optional

from app.core.config import settings
from app.core.logger import logger
from app.services.deepgram_stt_service import deepgram_stt_service


InterimCallback = Callable[[str, float], Awaitable[None]]
FinalCallback = Callable[[str, float], Awaitable[None]]


class SttPipeline:
    """
    Thin wrapper around the Deepgram streaming STT session.
    Responsible for:
    - Managing the underlying streaming session lifecycle.
    - Feeding MULAW audio bytes.
    - Invoking interim/final callbacks with transcript + confidence.
    """

    def __init__(
        self,
        language_code: Optional[str],
        on_interim: InterimCallback,
        on_final: FinalCallback,
        call_session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        endpointing_ms: Optional[int] = None,
    ) -> None:
        self._language_code = language_code
        self._on_interim = on_interim
        self._on_final = on_final
        self._call_session_id = call_session_id
        self._agent_id = agent_id
        # None → Deepgram uses settings.DEEPGRAM_STT_ENDPOINTING_MS when connecting
        self._endpointing_ms: Optional[int] = endpointing_ms

        self._stt_session = None
        self._reader_task: Optional[asyncio.Task] = None

    def _effective_endpointing_ms(self) -> int:
        if self._endpointing_ms is not None:
            return int(self._endpointing_ms)
        return int(getattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS", 900) or 900)

    async def _ensure_session(self) -> None:
        if self._stt_session is not None:
            return

        self._stt_session = deepgram_stt_service.create_streaming_session(
            language_code=self._language_code,
            encoding="MULAW",
            sample_rate=8000,
            interim_results=True,
            single_utterance=False,
            endpointing_ms=self._endpointing_ms,
        )

        async def consume_results():
            try:
                # Start underlying blocking stream in executor
                await self._stt_session.start()
            except Exception as e:
                logger.error(f"[STT] session start error: {e}", exc_info=True)

        async def reader_loop():
            while True:
                sess = self._stt_session
                if sess is None:
                    break
                try:
                    result = await sess.get_result()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"[STT] reader loop error: {e}", exc_info=True)
                    if self._stt_session is None:
                        break
                    continue

                if not result:
                    continue
                if result.get("done"):
                    break
                if result.get("error"):
                    logger.warning(
                        "[STT] session reported error payload: %s (call_session_id=%s, agent_id=%s)",
                        result.get("error"),
                        self._call_session_id,
                        self._agent_id,
                    )
                    continue

                transcript = (result.get("transcript") or "").strip()
                if not transcript:
                    continue

                is_final = bool(result.get("is_final"))
                confidence = float(result.get("confidence") or 0.0)

                try:
                    if is_final:
                        await self._on_final(transcript, confidence)
                    else:
                        await self._on_interim(transcript, confidence)
                except Exception as cb_err:
                    logger.error(f"[STT] callback error: {cb_err}", exc_info=True)

        # Kick off background readers
        self._reader_task = asyncio.create_task(reader_loop())
        asyncio.create_task(consume_results())

    async def feed_audio_chunk(self, audio_data: bytes) -> None:
        """
        Feed a raw MULAW audio chunk into the streaming session.
        Lazily starts the session on first call.
        """
        if not audio_data:
            return

        await self._ensure_session()
        if self._stt_session:
            self._stt_session.push_audio(audio_data)

    async def recreate_with_endpointing(self, endpointing_ms: int) -> None:
        """
        Close the current Deepgram session and reopen with a new endpointing (ms) value.
        Used after the agent asks for email so spelling pauses are less likely to split finals.
        """
        want = int(endpointing_ms)
        if want == self._effective_endpointing_ms() and self._stt_session is not None:
            return
        await self.aclose()
        self._endpointing_ms = want
        self._stt_session = None
        self._reader_task = None
        logger.info(
            "[STT] recreated session with endpointing_ms=%s (call_session_id=%s)",
            want,
            self._call_session_id,
        )

    def finish_session(self) -> None:
        """Signal the underlying STT session to finish.

        Only pushes the sentinel (None) onto the audio queue so the sender
        thread calls send_finalize / send_close_stream / socket.close and lets
        start_listening() unblock naturally.  Do NOT cancel the reader task here:
        the reader must stay alive to consume the final {"done": True} message
        that the Deepgram thread emits after the socket closes.  Cancelling it
        early strands the worker thread and keeps the Deepgram connection busy.
        """
        try:
            if self._stt_session:
                self._stt_session.finish()
        except Exception:
            pass

    async def aclose(self) -> None:
        """Async-friendly shutdown: signal finish then wait for reader to exit.

        Flow:
        1. finish_session() → pushes None onto audio queue.
        2. sender_loop sees None → calls _close_connection (finalize + close_stream
           + socket.close).
        3. start_listening() unblocks, emits {"done": True} → reader_loop breaks.
        4. We wait up to 5 s for the reader to complete cleanly.
        5. Only if the reader is still alive after the timeout do we cancel it
           as a last resort (prevents resource leak on hung connections).
        """
        self.finish_session()
        if self._reader_task and not self._reader_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._reader_task), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("[STT] reader_loop did not finish within 5 s — cancelling")
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except (asyncio.CancelledError, Exception):
                    pass
            except asyncio.CancelledError:
                pass
        self._stt_session = None
        self._reader_task = None

