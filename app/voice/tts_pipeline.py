import asyncio
from typing import Any, Dict, Optional

from app.core.logger import logger


class TtsPipeline:
    """
    TTS pipeline wrapper around the queue/worker pattern.

    Keeps:
    - A queue of TTS tasks (text + flags).
    - A background worker that asks the handler to stream each chunk.

    The heavy lifting of crossfade / jitter-buffer priming / background
    mixing still lives on the handler via its `_stream_tts_chunk` method.
    This keeps behaviour identical while making responsibilities clearer.
    """

    def __init__(self, handler: Any) -> None:
        # We deliberately keep a narrow dependency on the handler:
        # - handler._tts_cancel: asyncio.Event
        # - handler._stream_tts_chunk(text, use_ssml, is_final)
        # - handler._end_call_after_agent_request()
        self._handler = handler
        self._queue: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = asyncio.create_task(
            self._worker()
        )

    @property
    def cancel_event(self) -> asyncio.Event:
        return self._handler._tts_cancel  # type: ignore[attr-defined]

    @property
    def is_speaking(self) -> bool:
        # Source of truth still lives on the handler for now.
        return bool(getattr(self._handler, "is_speaking", False))

    async def queue_tts(self, task: Dict[str, Any]) -> None:
        """
        Enqueue a TTS task with fields:
        - text (str)
        - use_ssml (bool)
        - is_final (bool)
        - end_call_after (bool, optional)
        """
        await self._queue.put(task)

    async def cancel_current_and_clear_queue(self) -> None:
        """
        Barge-in helper:
        - Set cancel flag so current streaming stops.
        - Clear any queued-but-not-started TTS tasks.
        """
        if not self.cancel_event.is_set():
            self.cancel_event.set()

        # Clear queue
        try:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                self._queue.task_done()
        except asyncio.QueueEmpty:
            pass

        # Mark the handler as no longer speaking
        if hasattr(self._handler, "is_speaking"):
            self._handler.is_speaking = False

    async def shutdown(self) -> None:
        """
        Gracefully stop the worker.
        """
        try:
            if self._worker_task:
                await self._queue.put(None)
                await asyncio.wait_for(self._worker_task, timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            pass

    async def _worker(self) -> None:
        """
        Background worker for parallel TTS pipeline (Vapi-style).

        Pulls tasks from the queue and delegates to the handler's
        `_stream_tts_chunk` implementation for actual audio streaming.
        """
        try:
            while True:
                task = await self._queue.get()

                # Shutdown signal
                if task is None:
                    break

                # Check if cancelled (barge-in)
                if self.cancel_event.is_set():
                    self._queue.task_done()
                    continue

                try:
                    text = task.get("text", "")
                    use_ssml = task.get("use_ssml", False)
                    is_final = task.get("is_final", False)
                    end_call_after = task.get("end_call_after", False)

                    if not text or not text.strip():
                        self._queue.task_done()
                        continue

                    # Delegate the heavy lifting
                    await self._handler._stream_tts_chunk(  # type: ignore[attr-defined]
                        text,
                        use_ssml=use_ssml,
                        is_final=is_final,
                    )

                    # If agent response contained [END_CALL], end call after this TTS has played
                    if end_call_after and hasattr(
                        self._handler, "_end_call_after_agent_request"
                    ):
                        await self._handler._end_call_after_agent_request()
                except Exception as e:
                    logger.error(f"[TTS] pipeline worker loop error: {e}", exc_info=True)
                finally:
                    self._queue.task_done()
        except Exception as e:
            logger.error(f"[TTS] pipeline worker error: {e}", exc_info=True)

