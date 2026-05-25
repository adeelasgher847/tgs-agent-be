"""
Telephony router — /api/v1/telephony

Sprint 2 endpoints:
  POST /external   Register a BYO/SIP external number
  POST /bind       Bind a phone number to an agent (agent → ready)
  POST /unbind     Unbind a phone number from its agent (agent → pending)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.phone_number import (
    BindingStatusResponse,
    BindNumberRequest,
    RegisterExternalNumberRequest,
    RegisterExternalNumberResponse,
    UnbindNumberRequest,
)
from app.services.phone_number_service import phone_number_service
from app.utils.response import create_success_response

router = APIRouter()


@router.post("/external", response_model=SuccessResponse[RegisterExternalNumberResponse])
async def register_external_number(
    request: RegisterExternalNumberRequest,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> SuccessResponse[RegisterExternalNumberResponse]:
    """
    Register a BYO (bring-your-own) number via SIP.

    Stores provider='external', sip_username, sip_password (encrypted at rest).
    """
    pn = phone_number_service.register_external_number(
        db=db,
        phone_number=request.phone_number,
        tenant_id=user.current_tenant_id,
        sip_username=request.sip_username,
        sip_password=request.sip_password,
        label=request.label,
    )
    return create_success_response(
        RegisterExternalNumberResponse(
            id=pn.id,
            phone_number=pn.phone_number,
            provider="external",
            status=pn.status,
            workspace_id=pn.tenant_id,
            created_at=pn.created_at,
            message="External number registered successfully",
        ),
        "External number registered",
    )


@router.post("/bind", response_model=SuccessResponse[BindingStatusResponse])
async def bind_number(
    request: BindNumberRequest,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> SuccessResponse[BindingStatusResponse]:
    """
    Bind a phone number to an agent.

    - Sets phone_numbers.assistant_id (exposed as agent_id in API).
    - Sets agent.status = 'ready'.
    - Returns 409 if number is already bound.
    """
    pn = phone_number_service.bind_number(
        db=db,
        phone_number_id=request.number_id,
        agent_id=request.agent_id,
        tenant_id=user.current_tenant_id,
    )
    return create_success_response(
        BindingStatusResponse(
            number_id=pn.id,
            phone_number=pn.phone_number,
            agent_id=pn.assistant_id,
            agent_name=None,  # populated by list endpoint; omitted here for speed
            agent_status="ready",
            message=f"Number {pn.phone_number} bound to agent {request.agent_id}",
        ),
        "Number bound successfully",
    )


@router.post("/unbind", response_model=SuccessResponse[BindingStatusResponse])
async def unbind_number(
    request: UnbindNumberRequest,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> SuccessResponse[BindingStatusResponse]:
    """
    Remove an agent binding from a phone number.

    - Clears phone_numbers.assistant_id.
    - Sets agent.status = 'pending'.
    - Returns 409 if number is not currently bound.
    """
    pn = phone_number_service.unbind_number(
        db=db,
        phone_number_id=request.number_id,
        tenant_id=user.current_tenant_id,
    )
    return create_success_response(
        BindingStatusResponse(
            number_id=pn.id,
            phone_number=pn.phone_number,
            agent_id=None,
            agent_name=None,
            agent_status="pending",
            message=f"Number {pn.phone_number} unbound; agent set to pending",
        ),
        "Number unbound successfully",
    )
