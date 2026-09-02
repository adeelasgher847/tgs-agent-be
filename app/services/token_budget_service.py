"""
Per-tenant daily LLM token budget tracking.

Usage counters live entirely in Redis (fixed-window per UTC day, 48h TTL) so
that recording a turn's token usage never touches the DB on the hot call
path. ``Tenant.llm_token_budget_daily`` (nullable) is the only piece of
state that lives in Postgres — ``NULL`` means "observe only, no cap".

Every entrypoint here fails open: a Redis outage, DB timeout, or any other
exception must never block or fail a live call turn or a usage read. On
failure we log a warning and return conservative defaults (zero usage, no
budget, "not exceeded").

KNOWN GAP: wired into BidirectionalStreamHandler.generate_and_stream_response
and ConversationOrchestrator.generate_and_stream_response — i.e. the
standard STT -> LLM -> TTS turn path only. Gemini Live / OpenAI Realtime
native-audio calls bypass that method entirely (see the native-audio forks
in app/voice/voice_orchestrator.py) and are therefore NOT covered: no
budget enforcement, no usage recording. Deliberately out of scope for this
phase — those models don't expose separate prompt/response text to run the
token estimate against, so covering them needs its own design.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.tenant import Tenant
from app.utils.redis_client import get_redis

# Counters are kept for 48h so a usage read shortly after UTC midnight can
# still see "yesterday's" totals without needing a DB round-trip.
_COUNTER_TTL_SECONDS = 172800


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _tokens_key(tenant_id: uuid.UUID, date_str: str) -> str:
    return f"tenant:{tenant_id}:tokens:{date_str}"


def _calls_key(tenant_id: uuid.UUID, date_str: str) -> str:
    return f"tenant:{tenant_id}:calls:{date_str}"


class TokenBudgetService:
    """Tracks and enforces per-tenant daily LLM token budgets."""

    async def record_daily_tokens(self, tenant_id: uuid.UUID, tokens: int) -> int:
        """Atomically increment today's token/call counters for a tenant.

        Returns the updated daily token total, or 0 if Redis is unavailable
        or the operation failed for any reason — this must never block or
        fail a live call turn.
        """
        redis_client = get_redis()
        if redis_client is None:
            return 0

        date_str = _today_utc()
        tokens_key = _tokens_key(tenant_id, date_str)
        calls_key = _calls_key(tenant_id, date_str)

        try:
            pipe = redis_client.pipeline()
            pipe.incrby(tokens_key, tokens)
            pipe.expire(tokens_key, _COUNTER_TTL_SECONDS)
            pipe.incr(calls_key)
            pipe.expire(calls_key, _COUNTER_TTL_SECONDS)
            results = await pipe.execute()
            return int(results[0])
        except Exception:
            logger.warning(
                "token_budget_service: failed to record daily tokens for tenant=%s",
                tenant_id,
                exc_info=True,
            )
            return 0

    def _get_budget_limit_sync(self, db: Session, tenant_id: uuid.UUID) -> int | None:
        try:
            tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
            return tenant.llm_token_budget_daily if tenant else None
        except Exception:
            logger.warning(
                "token_budget_service: failed to look up budget limit for tenant=%s",
                tenant_id,
                exc_info=True,
            )
            return None

    async def _get_budget_limit(self, db: Session, tenant_id: uuid.UUID) -> int | None:
        # Dispatched off the event loop via asyncio.to_thread — this is a
        # synchronous Session.query() and this method runs on every real
        # conversational turn (called from check_daily_budget), so calling
        # it in-loop would risk stalling every concurrent live call sharing
        # this asyncio worker on a Postgres latency spike. Same fix pattern
        # as credit_service's PR #246 event-loop-blocking fix.
        return await asyncio.to_thread(self._get_budget_limit_sync, db, tenant_id)

    async def _get_usage_counts(self, tenant_id: uuid.UUID, date_str: str) -> tuple[int, int]:
        redis_client = get_redis()
        if redis_client is None:
            return 0, 0

        try:
            tokens_raw = await redis_client.get(_tokens_key(tenant_id, date_str))
            calls_raw = await redis_client.get(_calls_key(tenant_id, date_str))
            total_tokens = int(tokens_raw) if tokens_raw is not None else 0
            total_calls = int(calls_raw) if calls_raw is not None else 0
            return total_tokens, total_calls
        except Exception:
            logger.warning(
                "token_budget_service: failed to read usage counters for tenant=%s",
                tenant_id,
                exc_info=True,
            )
            return 0, 0

    async def get_daily_usage(
        self, db: Session, tenant_id: uuid.UUID, date_str: str | None = None
    ) -> dict:
        """Return today's (or a given day's) token/call usage plus budget status.

        Fails open: never raises. Best-effort defaults are returned if Redis
        or the DB are unavailable.
        """
        resolved_date = date_str or _today_utc()

        total_tokens, total_calls = await self._get_usage_counts(tenant_id, resolved_date)
        budget_limit = await self._get_budget_limit(db, tenant_id)

        is_exceeded = budget_limit is not None and total_tokens >= budget_limit

        return {
            "tenant_id": str(tenant_id),
            "date": resolved_date,
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "budget_limit": budget_limit,
            "is_exceeded": is_exceeded,
        }

    async def check_daily_budget(
        self, db: Session, tenant_id: uuid.UUID
    ) -> tuple[bool, int, int | None]:
        """Check whether a tenant is within its daily LLM token budget.

        Returns ``(within_budget, current_usage, budget_limit)``. Fails open
        on any Redis/DB error — a broken budget check must never block a
        call, so failures return ``(True, 0, None)``.
        """
        try:
            date_str = _today_utc()
            budget_limit = await self._get_budget_limit(db, tenant_id)
            current_usage, _ = await self._get_usage_counts(tenant_id, date_str)

            if budget_limit is None:
                return True, current_usage, None

            return current_usage < budget_limit, current_usage, budget_limit
        except Exception:
            logger.warning(
                "token_budget_service: budget check failed for tenant=%s (failing open)",
                tenant_id,
                exc_info=True,
            )
            return True, 0, None


token_budget_service = TokenBudgetService()
