"""
v2 Smart Callback Scheduler endpoints.

Auth: any authenticated tenant principal (API key or JWT).

PUT  /api/v2/agents/{agent_id}/callback-config
GET  /api/v2/agents/{agent_id}/callback-status
GET  /api/v2/calls/{call_id}/callback-history
GET  /api/v2/calls/{call_id}/memory-context
"""
from __future__ import annotations

import uuid
from typing import List, Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.core.request_auth import ApiKeyPrincipal
from app.models.call_session import CallSession
from app.models.user import User
from app.schemas.callback_scheduler import (
    CallbackConfigResponse,
    CallbackConfigUpdate,
    CallbackHistoryItem,
    CallbackStatusResponse,
)
from app.services.callback_scheduler_service import callback_scheduler_service


class CallMemoryContextResponse(BaseModel):
    memory_context: str


def _tenant_id(principal: Union[User, ApiKeyPrincipal]) -> uuid.UUID:
    return principal.current_tenant_id


# ── /api/v2/agents/… ──────────────────────────────────────────────────────────

agents_router = APIRouter(prefix="/agents", tags=["Smart Callback Scheduler"])


@agents_router.put(
    "/{agent_id}/callback-config",
    response_model=CallbackConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Update agent smart-callback configuration",
)
def update_callback_config(
    agent_id: uuid.UUID,
    payload: CallbackConfigUpdate,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> CallbackConfigResponse:
    """
    Set the smart-callback configuration for an agent.

    - **smart_callback_enabled**: activates the retry loop.
    - **max_attempts**: total retry budget (1–20).
    - **gap_schedule**: ordered list of `{days, hours}` gaps between attempts.
    - **timezone**: IANA timezone string — returns 422 if invalid.
    """
    return callback_scheduler_service.update_callback_config(
        db, agent_id, _tenant_id(principal), payload
    )


@agents_router.get(
    "/{agent_id}/callback-status",
    response_model=CallbackStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get agent callback config and live retry counters",
)
def get_callback_status(
    agent_id: uuid.UUID,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> CallbackStatusResponse:
    """
    Returns the agent's callback config plus:
    - `pending_retries`: number of pending callback records.
    - `next_scheduled_at`: UTC time of the earliest pending callback.
    """
    return callback_scheduler_service.get_callback_status(
        db, agent_id, _tenant_id(principal)
    )


# ── /api/v2/calls/… ───────────────────────────────────────────────────────────

calls_router = APIRouter(prefix="/calls", tags=["Smart Callback Scheduler"])


@calls_router.get(
    "/{call_id}/callback-history",
    response_model=List[CallbackHistoryItem],
    status_code=status.HTTP_200_OK,
    summary="Get callback retry history for a call",
)
def get_callback_history(
    call_id: uuid.UUID,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> List[CallbackHistoryItem]:
    """
    Returns ordered callback attempts for `call_id`.
    Returns 404 if the call does not belong to the caller's tenant.
    """
    return callback_scheduler_service.get_callback_history(
        db, call_id, _tenant_id(principal)
    )


@calls_router.get(
    "/{call_id}/memory-context",
    response_model=CallMemoryContextResponse,
    status_code=status.HTTP_200_OK,
    summary="Debug: view the cached cross-session caller memory context for a call",
)
def get_call_memory_context(
    call_id: uuid.UUID,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> CallMemoryContextResponse:
    """
    Returns the caller-memory context block cached on the call's
    `call_metadata["caller_memory_context"]` at prompt-build time.
    Returns 404 if the call does not belong to the caller's tenant.
    """
    tenant_id = principal.current_tenant_id
    call_session = db.execute(
        select(CallSession).where(
            CallSession.id == call_id,
            CallSession.tenant_id == tenant_id,
        )
    ).scalar_one_or_none()

    if call_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call session {call_id} not found",
        )

    metadata = call_session.call_metadata or {}
    return CallMemoryContextResponse(memory_context=metadata.get("caller_memory_context") or "")
