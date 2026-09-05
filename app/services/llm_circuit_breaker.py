"""Redis-backed circuit breaker for LLM provider calls in the voice hot path.

Live voice turns cannot tolerate blocking on a dead LLM provider — each turn
would otherwise wait out the full ``VOICE_TURN_TIMEOUT_SEC`` (15-20s) of dead
air. This breaker tracks consecutive failures per provider in Redis (shared
across every call/worker process), trips OPEN after
``LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD`` consecutive failures, and allows a
single probe call through once ``LLM_CIRCUIT_BREAKER_COOLDOWN_SEC`` has
elapsed (HALF_OPEN) to test recovery before fully closing again.

Fails open on any Redis error or when the breaker is disabled in settings —
this must never block or fail a live call turn.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.core.config import settings
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"

_KEY_PREFIX = "circuit:llm"


def _state_key(provider: str) -> str:
    return f"{_KEY_PREFIX}:{provider}:state"


def _failures_key(provider: str) -> str:
    return f"{_KEY_PREFIX}:{provider}:failures"


def _cooldown_key(provider: str) -> str:
    return f"{_KEY_PREFIX}:{provider}:cooldown"


class LLMCircuitBreaker:
    """Distributed circuit breaker for LLM providers (openai, gemini, ...).

    Non-blocking: every method fails open (returns True / no-ops) on any
    Redis error rather than risking dead air on a live call.
    """

    async def can_execute(self, provider: str) -> bool:
        """Return True if a call to ``provider`` should proceed right now."""
        if not settings.llm.circuit_breaker_enabled:
            return True

        redis_client = get_redis()
        if redis_client is None:
            return True

        try:
            state = await redis_client.get(_state_key(provider))
            state = state or STATE_CLOSED

            if state == STATE_CLOSED:
                return True

            # OPEN or HALF_OPEN: the cooldown key's TTL is the single source
            # of truth for whether a probe is allowed through. While it
            # exists, either the cooldown hasn't elapsed yet (OPEN) or a
            # probe is already in flight (HALF_OPEN) — either way, fast-fail.
            # Once it's gone (elapsed, or a crashed probe never resolved it),
            # SET NX atomically grants exactly one caller the probe slot.
            acquired = await redis_client.set(
                _cooldown_key(provider),
                "probe",
                nx=True,
                ex=settings.llm.circuit_breaker_cooldown_sec,
            )
            if acquired:
                await redis_client.set(_state_key(provider), STATE_HALF_OPEN)
                return True
            return False
        except Exception:  # noqa: BLE001 - fail open, never block a live turn
            logger.warning(
                "[CircuitBreaker] Redis error checking state for %s — failing open",
                provider,
                exc_info=True,
            )
            return True

    async def record_success(self, provider: str) -> None:
        """Reset the breaker to CLOSED after a successful provider call."""
        redis_client = get_redis()
        if redis_client is None:
            return

        try:
            await redis_client.set(_failures_key(provider), 0)
            await redis_client.set(_state_key(provider), STATE_CLOSED)
            await redis_client.delete(_cooldown_key(provider))
        except Exception:  # noqa: BLE001 - fail open
            logger.warning(
                "[CircuitBreaker] Redis error recording success for %s",
                provider,
                exc_info=True,
            )

    async def record_failure(
        self, provider: str, error: Optional[Exception] = None
    ) -> None:
        """Record a provider failure; trip the breaker OPEN past threshold."""
        redis_client = get_redis()
        if redis_client is None:
            return

        try:
            failures = await redis_client.incr(_failures_key(provider))
            threshold = settings.llm.circuit_breaker_failure_threshold
            if failures >= threshold:
                await redis_client.set(_state_key(provider), STATE_OPEN)
                await redis_client.set(
                    _cooldown_key(provider),
                    "cooling_down",
                    ex=settings.llm.circuit_breaker_cooldown_sec,
                )
                logger.warning(
                    "[CircuitBreaker] %s tripped OPEN after %d consecutive failures: %s",
                    provider,
                    failures,
                    error,
                )
        except Exception:  # noqa: BLE001 - fail open
            logger.warning(
                "[CircuitBreaker] Redis error recording failure for %s",
                provider,
                exc_info=True,
            )


llm_circuit_breaker = LLMCircuitBreaker()
