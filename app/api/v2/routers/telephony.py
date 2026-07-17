"""
v2 Telephony — outbound number reputation monitoring.

Auth: API key + x-workspace-id, or JWT dashboard user (get_workspace + role gate).

POST /api/v2/telephony/check-reputation    — on-demand reputation check for one number
GET  /api/v2/telephony/reputation-summary  — reputation snapshot for every workspace number
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_workspace
from app.api.deps.rbac import require_config_or_api_key, require_readonly_or_api_key
from app.core.workspace import Workspace
from app.models.phone_number import PhoneNumber
from app.models.phone_number_reputation import PhoneNumberReputation
from app.services.reputation_service import check_number_reputation

router = APIRouter(prefix="/telephony", tags=["Telephony"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class CheckReputationRequest(BaseModel):
    phone_number_id: uuid.UUID


class ReputationResult(BaseModel):
    spam_flagged: bool
    reputation_score: int
    flagged_reason: Optional[str] = None


class ReputationSummaryItem(BaseModel):
    phone_number: str
    reputation_score: int
    spam_flagged: bool
    last_checked_at: Optional[datetime] = None


# ── POST /telephony/check-reputation ────────────────────────────────────────

@router.post("/check-reputation", response_model=ReputationResult)
async def check_reputation(
    body: CheckReputationRequest,
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    _principal=Depends(require_config_or_api_key),
) -> ReputationResult:
    """Run a reputation check for a single workspace phone number and persist the result."""
    phone_number_obj = (
        db.query(PhoneNumber)
        .filter(
            PhoneNumber.id == body.phone_number_id,
            PhoneNumber.tenant_id == workspace.id,
        )
        .first()
    )
    if phone_number_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Phone number not found in this workspace",
        )

    result = await check_number_reputation(db, phone_number_obj)
    return ReputationResult(**result)


# ── GET /telephony/reputation-summary ───────────────────────────────────────

@router.get("/reputation-summary", response_model=List[ReputationSummaryItem])
async def reputation_summary(
    workspace: Workspace = Depends(get_workspace),
    db: Session = Depends(get_db),
    _principal=Depends(require_readonly_or_api_key),
) -> List[ReputationSummaryItem]:
    """Reputation snapshot for every phone number owned by this workspace."""
    rows = db.execute(
        select(PhoneNumber, PhoneNumberReputation)
        .outerjoin(
            PhoneNumberReputation,
            PhoneNumberReputation.phone_number_id == PhoneNumber.id,
        )
        .where(PhoneNumber.tenant_id == workspace.id)
    ).all()

    return [
        ReputationSummaryItem(
            phone_number=phone_number.phone_number,
            reputation_score=reputation.reputation_score if reputation else 100,
            spam_flagged=reputation.spam_flagged if reputation else False,
            last_checked_at=reputation.last_checked_at if reputation else None,
        )
        for phone_number, reputation in rows
    ]
