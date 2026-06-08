"""
Typed STT event definitions and asyncio-based event bus.

Events emitted by SttPipeline providers:
  SttInterimEvent  — partial transcript, updated as speech continues
  SttFinalEvent    — confirmed utterance, may carry isSilence=True on pause
  SttErrorEvent    — provider error with recoverable flag

The SttEventBus allows multiple async subscribers (e.g. VoiceOrchestrator)
to receive events without coupling to the callback signature.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Union


# ── Event types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SttInterimEvent:
    type: Literal["interim"] = field(default="interim", init=False)
    transcript: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class SttFinalEvent:
    type: Literal["final"] = field(default="final", init=False)
    transcript: str = ""
    confidence: float = 0.0
    is_silence: bool = False


@dataclass(frozen=True)
class SttErrorEvent:
    type: Literal["error"] = field(default="error", init=False)
    message: str = ""
    recoverable: bool = True


SttEvent = Union[SttInterimEvent, SttFinalEvent, SttErrorEvent]

SttEventCallback = Callable[[SttEvent], Awaitable[None]]


# ── Event bus ─────────────────────────────────────────────────────────────────

class SttEventBus:
    """Lightweight asyncio pub/sub for STT events.

    VoiceOrchestrator subscribes once; the provider emits via emit().
    Subscribers are called sequentially in subscription order.
    Errors in one subscriber are logged and do not block subsequent ones.
    """

    def __init__(self) -> None:
        self._subscribers: list[SttEventCallback] = []

    def subscribe(self, callback: SttEventCallback) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: SttEventCallback) -> None:
        self._subscribers = [s for s in self._subscribers if s is not callback]

    async def emit(self, event: SttEvent) -> None:
        for cb in list(self._subscribers):
            try:
                await cb(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                from app.core.logger import logger
                logger.error(
                    "[SttEventBus] subscriber error on %s event",
                    event.type,
                    exc_info=True,
                )
