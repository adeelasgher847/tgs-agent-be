"""
CancellationToken: Async-safe cancellation mechanism for voice orchestration.

Enables atomic cancellation of all running tasks on barge-in or call shutdown.
Prevents task orphaning and ensures clean interruption cascade.
"""

import asyncio
from contextvars import ContextVar
from typing import Set
import logging

logger = logging.getLogger(__name__)

# Context variable for cancellation scope
cancellation_context: ContextVar['CancellationToken'] = ContextVar(
    'cancellation_token',
    default=None
)


class CancellationToken:
    """
    Thread-safe, async-safe cancellation token for voice orchestration.
    
    Usage:
        token = CancellationToken()
        
        # Register task for tracking
        task = asyncio.create_task(some_coro())
        await token.register_task(task)
        
        # Check cancellation frequently
        if token.is_cancelled():
            break
            
        # Cancel all tasks
        await token.cancel_all()
    """

    def __init__(self, call_id: str = "unknown"):
        self.call_id = call_id
        self._is_cancelled = False
        self._tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def is_cancelled(self) -> bool:
        """
        Check if cancellation was requested.
        
        Call this frequently in loops to react to barge-in/shutdown.
        """
        return self._is_cancelled

    async def register_task(self, task: asyncio.Task) -> None:
        """
        Register task for tracking and cancellation.
        
        Must be called immediately after task creation to ensure
        it's cancelled when cancel_all() is triggered.
        """
        async with self._lock:
            self._tasks.add(task)

        # Auto-remove task from registry when it completes
        task.add_done_callback(lambda t: self._tasks.discard(t))

    async def cancel_all(self, timeout_ms: int = 100) -> None:
        """
        Cancel ALL registered tasks immediately.
        
        Called on:
        - Barge-in detection (user interrupts agent)
        - Call shutdown
        - Fatal errors
        
        Args:
            timeout_ms: Max time to wait for tasks to finish (default: 100ms for <100ms barge-in)
        """
        self._is_cancelled = True

        # Get snapshot of tasks to cancel
        async with self._lock:
            tasks_to_cancel = list(self._tasks)

        if not tasks_to_cancel:
            logger.debug(f"[{self.call_id}] No tasks to cancel")
            self._is_cancelled = False  # Always reset, even with no tasks
            return

        logger.info(
            f"[{self.call_id}] Cancelling {len(tasks_to_cancel)} tasks "
            f"(timeout: {timeout_ms}ms)"
        )

        # Cancel all tasks
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()

        # Wait for all tasks to finish (with timeout)
        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[{self.call_id}] Task cancellation timed out after {timeout_ms}ms. "
                    f"Forcing cancellation."
                )
                # Force kill any stragglers
                for task in tasks_to_cancel:
                    if not task.done():
                        task.cancel()

        async with self._lock:
            self._tasks.clear()

        # Reset flag for next utterance
        self._is_cancelled = False
        logger.debug(f"[{self.call_id}] Cancellation complete")

    async def __aenter__(self):
        """Context manager: set token scope"""
        cancellation_context.set(self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager: cleanup on exit"""
        cancellation_context.set(None)
        return False
