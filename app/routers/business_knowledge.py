from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_owner
from app.models.agent import Agent
from app.models.business_knowledge import BusinessKnowledge
from app.schemas.base import SuccessResponse
from app.schemas.business_knowledge import (
    BusinessKnowledgeCreate,
    BusinessKnowledgeList,
    BusinessKnowledgeOut,
    BusinessKnowledgeUpdate,
)
from app.utils.response import create_success_response

router = APIRouter()


def _get_record_or_404(
    db: Session, record_id: uuid.UUID, tenant_id: uuid.UUID
) -> BusinessKnowledge:
    record = (
        db.query(BusinessKnowledge)
        .filter(
            BusinessKnowledge.id == record_id,
            BusinessKnowledge.tenant_id == tenant_id,
            BusinessKnowledge.is_active == True,  # noqa: E712
        )
        .first()
    )
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Business knowledge record not found.",
        )
    return record


@router.post(
    "",
    response_model=SuccessResponse[BusinessKnowledgeOut],
    status_code=status.HTTP_201_CREATED,
)
def create_business_knowledge(
    payload: BusinessKnowledgeCreate,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    tenant_id: uuid.UUID = user.current_tenant_id

    if payload.agent_id is not None:
        agent = (
            db.query(Agent)
            .filter(
                Agent.id == payload.agent_id,
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,  # noqa: E712
            )
            .first()
        )
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="agent_id not found or does not belong to your tenant.",
            )

    record = BusinessKnowledge(
        tenant_id=tenant_id,
        **payload.model_dump(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return create_success_response(
        BusinessKnowledgeOut.model_validate(record),
        "Business knowledge record created successfully.",
    )


@router.get(
    "",
    response_model=SuccessResponse[BusinessKnowledgeList],
)
def list_business_knowledge(
    agent_id: Optional[uuid.UUID] = None,
    include_inactive: bool = False,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    tenant_id: uuid.UUID = user.current_tenant_id

    query = db.query(BusinessKnowledge).filter(
        BusinessKnowledge.tenant_id == tenant_id
    )

    if agent_id is not None:
        query = query.filter(BusinessKnowledge.agent_id == agent_id)

    if not include_inactive:
        query = query.filter(BusinessKnowledge.is_active == True)  # noqa: E712

    records = query.order_by(BusinessKnowledge.created_at.asc()).all()

    return create_success_response(
        BusinessKnowledgeList(
            items=[BusinessKnowledgeOut.model_validate(r) for r in records],
            total=len(records),
        ),
        "Business knowledge records fetched successfully.",
    )


@router.get(
    "/{record_id}",
    response_model=SuccessResponse[BusinessKnowledgeOut],
)
def get_business_knowledge(
    record_id: uuid.UUID,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    tenant_id: uuid.UUID = user.current_tenant_id
    record = _get_record_or_404(db, record_id, tenant_id)
    return create_success_response(
        BusinessKnowledgeOut.model_validate(record),
        "Business knowledge record fetched successfully.",
    )


@router.patch(
    "/{record_id}",
    response_model=SuccessResponse[BusinessKnowledgeOut],
)
def update_business_knowledge(
    record_id: uuid.UUID,
    payload: BusinessKnowledgeUpdate,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    tenant_id: uuid.UUID = user.current_tenant_id
    record = _get_record_or_404(db, record_id, tenant_id)

    update_data = payload.model_dump(exclude_unset=True)

    if "agent_id" in update_data and update_data["agent_id"] is not None:
        agent = (
            db.query(Agent)
            .filter(
                Agent.id == update_data["agent_id"],
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,  # noqa: E712
            )
            .first()
        )
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="agent_id not found or does not belong to your tenant.",
            )

    for field, value in update_data.items():
        setattr(record, field, value)

    db.commit()
    db.refresh(record)

    return create_success_response(
        BusinessKnowledgeOut.model_validate(record),
        "Business knowledge record updated successfully.",
    )


@router.delete(
    "/{record_id}",
    response_model=SuccessResponse[dict],
)
def delete_business_knowledge(
    record_id: uuid.UUID,
    user=Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    tenant_id: uuid.UUID = user.current_tenant_id
    record = _get_record_or_404(db, record_id, tenant_id)

    record.is_active = False
    db.commit()

    return create_success_response(
        {"record_id": str(record_id)},
        "Business knowledge record deleted successfully.",
    )
