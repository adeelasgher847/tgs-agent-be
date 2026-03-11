import asyncio
from typing import Awaitable, Callable, Optional

from app.core.logger import logger
from app.services.google_stt_service import google_stt_service


InterimCallback = Callable[[str, float], Awaitable[None]]
FinalCallback = Callable[[str, float], Awaitable[None]]


class SttPipeline:
    """
    Thin wrapper around the Google streaming STT session.
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
    ) -> None:
        self._language_code = language_code
        self._on_interim = on_interim
        self._on_final = on_final

        self._stt_session = None
        self._reader_task: Optional[asyncio.Task] = None
        self._started = False

    async def _ensure_session(self) -> None:
        if self._stt_session is not None:
            return

        self._stt_session = google_stt_service.create_streaming_session(
            language_code=self._language_code,
            encoding="MULAW",
            sample_rate=8000,
            interim_results=True,
            single_utterance=False,
        )

        async def consume_results():
            try:
                # Start underlying blocking stream in executor
                await self._stt_session.start()
            except Exception as e:
                logger.error(f"STT session start error: {e}", exc_info=True)

        async def reader_loop():
            while True:
                try:
                    result = await self._stt_session.get_result()
                except Exception as e:
                    logger.error(f"STT reader loop error: {e}", exc_info=True)
                    continue

                if not result:
                    continue
                if result.get("error"):
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
                    logger.error(f"STT callback error: {cb_err}", exc_info=True)

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

    def finish_session(self) -> None:
        """
        Signal the underlying STT session to finish.
        """
        try:
            if self._stt_session:
                self._stt_session.finish()
        except Exception:
            # Never raise on shutdown path
            pass

