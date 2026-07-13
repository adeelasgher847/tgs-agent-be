"""Integration service — Make.com and n8n workspace-scoped helpers."""
from __future__ import annotations

import secrets
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logger import logger
from app.models.tenant import Tenant
from app.models.agent import Agent
from sqlalchemy.orm import Session

# Per-workspace integration rate limit: 10 calls per 60 seconds.
INTEGRATION_RATE_LIMIT = 10
INTEGRATION_RATE_WINDOW = 60  # seconds

_redis: Optional[aioredis.Redis] = None


def _get_redis() -> Optional[aioredis.Redis]:
    global _redis
    if _redis is None:
        try:
            _redis = aioredis.from_url(
                settings.REDIS_URL, encoding="utf-8", decode_responses=True
            )
        except Exception as exc:
            logger.warning("Integration service: Redis unavailable (%s) — skipping rate limit", exc)
    return _redis


async def check_integration_rate_limit(workspace_id: uuid.UUID) -> tuple[bool, float]:
    """
    Sliding-window rate limit: 10 integration-triggered calls per minute per workspace.
    Returns (allowed, retry_after_epoch_seconds).
    """
    r = _get_redis()
    if r is None:
        return True, 0.0

    now = time.time()
    window_start = now - INTEGRATION_RATE_WINDOW
    key = f"integration_rate:{workspace_id}"

    try:
        pipe = r.pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zadd(key, {f"{now}:{uuid.uuid4().hex}": now})
        pipe.zcard(key)
        pipe.zrange(key, 0, 0, withscores=True)
        pipe.expire(key, INTEGRATION_RATE_WINDOW)
        results = await pipe.execute()

        count = results[2]
        oldest = results[3]

        if count > INTEGRATION_RATE_LIMIT:
            await r.zremrangebyscore(key, now, now + 0.001)
            retry_after = (oldest[0][1] + INTEGRATION_RATE_WINDOW) if oldest else (now + INTEGRATION_RATE_WINDOW)
            return False, retry_after

        return True, 0.0
    except Exception as exc:
        logger.warning("Integration rate limit check failed: %s — allowing", exc)
        return True, 0.0


def generate_make_secret() -> str:
    """Generate a 32-byte hex secret for Make.com workspace integration."""
    return secrets.token_hex(32)


def generate_n8n_secret() -> str:
    """Generate a 32-byte hex secret for n8n workspace integration."""
    return secrets.token_hex(32)


def get_workspace_settings(tenant: Tenant) -> dict:
    return tenant.workspace_settings or {}


def store_make_secret(db: Session, tenant: Tenant, secret: str) -> None:
    settings_dict = dict(get_workspace_settings(tenant))
    settings_dict["make_secret"] = secret
    tenant.workspace_settings = settings_dict
    db.commit()
    db.refresh(tenant)


def get_make_secret(tenant: Tenant) -> Optional[str]:
    return get_workspace_settings(tenant).get("make_secret")


def store_n8n_secret(db: Session, tenant: Tenant, secret: str) -> None:
    settings_dict = dict(get_workspace_settings(tenant))
    settings_dict["n8n_secret"] = secret
    tenant.workspace_settings = settings_dict
    db.commit()
    db.refresh(tenant)


def get_n8n_secret(tenant: Tenant) -> Optional[str]:
    return get_workspace_settings(tenant).get("n8n_secret")


def record_last_triggered(db: Session, tenant: Tenant, integration: str) -> None:
    """Persist last_triggered_at for a named integration (make | n8n)."""
    settings_dict = dict(get_workspace_settings(tenant))
    settings_dict[f"{integration}_last_triggered_at"] = datetime.now(timezone.utc).isoformat()
    tenant.workspace_settings = settings_dict
    db.commit()


def get_last_triggered_at(tenant: Tenant, integration: str) -> Optional[datetime]:
    raw = get_workspace_settings(tenant).get(f"{integration}_last_triggered_at")
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def resolve_tenant_by_agent(db: Session, agent_id: str) -> tuple[Optional[Agent], Optional[Tenant]]:
    """Look up an agent and its owning tenant by agent_id string."""
    try:
        agent_uuid = uuid.UUID(agent_id)
    except ValueError:
        return None, None

    agent = db.query(Agent).filter(Agent.id == agent_uuid).first()
    if agent is None:
        return None, None

    tenant = db.query(Tenant).filter(Tenant.id == agent.tenant_id).first()
    return agent, tenant
