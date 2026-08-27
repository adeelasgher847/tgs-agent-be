"""Inbound Rules & Number Blocklist Rule Sets v2 API Router."""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from app.api.deps import (
    get_db,
    require_admin_or_api_key,
    require_readonly_or_api_key,
)
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User
from app.schemas.inbound_rule import (
    InboundRuleImportRequest,
    InboundRuleImportResponse,
    InboundRuleSetCreate,
    InboundRuleSetListItem,
    InboundRuleSetResponse,
    InboundRuleSetUpdate,
)
from app.services.audit_service import log_audit_event
from app.services.inbound_rules_service import inbound_rules_service

router = APIRouter(prefix="/inbound-rules", tags=["Inbound Rules & Blocklist"])


def _tenant_id(principal: User | ApiKeyPrincipal) -> uuid.UUID:
    return getattr(principal, "current_tenant_id", None) or getattr(
        principal, "tenant_id", None
    )


@router.get(
    "/sets",
    response_model=List[InboundRuleSetListItem],
    status_code=status.HTTP_200_OK,
    summary="List all Inbound Rule Sets for the workspace",
    description="Returns all non-deleted inbound rule sets with active rules count for current workspace. Read-only rank is sufficient.",
)
async def list_rule_sets(
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> List[InboundRuleSetListItem]:
    tenant_id = _tenant_id(principal)
    return inbound_rules_service.list_rule_sets(db, tenant_id)


@router.post(
    "/sets",
    response_model=InboundRuleSetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new Inbound Rule Set",
    description="Creates a new rule set and optionally initializes it with number rules. Admin rank or API key required.",
)
async def create_rule_set(
    body: InboundRuleSetCreate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> InboundRuleSetResponse:
    tenant_id = _tenant_id(principal)
    user_id = principal.id if isinstance(principal, User) else None
    result = inbound_rules_service.create_rule_set(
        db, tenant_id, user_id, body
    )
    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="inbound_rule_set.created",
        resource_type="inbound_rule_set",
        resource_id=result.id,
        new_value=result.model_dump(),
        actor_user_id=principal.id,
    )
    return result


@router.post(
    "/sets/import",
    response_model=InboundRuleImportResponse,
    status_code=status.HTTP_200_OK,
    summary="Bulk import number rules from CSV / text",
    description="Imports number rules from multiline text or CSV data with optional labels into an existing or new rule set. Admin rank or API key required.",
)
async def import_rules(
    body: InboundRuleImportRequest,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> InboundRuleImportResponse:
    tenant_id = _tenant_id(principal)
    user_id = principal.id if isinstance(principal, User) else None
    result = inbound_rules_service.import_rules_from_text(
        db, tenant_id, user_id, body
    )
    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="inbound_rule_set.imported",
        resource_type="inbound_rule_set",
        resource_id=result.rule_set.id,
        new_value=result.model_dump(),
        actor_user_id=principal.id,
    )
    return result


@router.get(
    "/sets/{set_id}",
    response_model=InboundRuleSetResponse,
    status_code=status.HTTP_200_OK,
    summary="Get single Inbound Rule Set with rules list",
    description="Returns rule set details and all its active number rules. Read-only rank is sufficient.",
)
async def get_rule_set(
    set_id: uuid.UUID,
    principal: User | ApiKeyPrincipal = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> InboundRuleSetResponse:
    tenant_id = _tenant_id(principal)
    return inbound_rules_service.get_rule_set(db, tenant_id, set_id)


@router.put(
    "/sets/{set_id}",
    response_model=InboundRuleSetResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Inbound Rule Set",
    description="Updates name/description and batch replaces number rules if rules array is provided. Admin rank or API key required.",
)
async def update_rule_set(
    set_id: uuid.UUID,
    body: InboundRuleSetUpdate,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> InboundRuleSetResponse:
    tenant_id = _tenant_id(principal)
    result = inbound_rules_service.update_rule_set(
        db, tenant_id, set_id, body
    )
    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="inbound_rule_set.updated",
        resource_type="inbound_rule_set",
        resource_id=result.id,
        new_value=result.model_dump(),
        actor_user_id=principal.id,
    )
    return result


@router.delete(
    "/sets/{set_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an Inbound Rule Set",
    description="Soft deletes the rule set and its rules, detaching it from any call flows. Admin rank or API key required.",
)
async def delete_rule_set(
    set_id: uuid.UUID,
    request: Request,
    principal: User | ApiKeyPrincipal = Depends(require_admin_or_api_key),
    db: Session = Depends(get_db),
) -> None:
    tenant_id = _tenant_id(principal)
    inbound_rules_service.delete_rule_set(db, tenant_id, set_id)
    log_audit_event(
        db,
        request=request,
        tenant_id=tenant_id,
        action="inbound_rule_set.deleted",
        resource_type="inbound_rule_set",
        resource_id=set_id,
        actor_user_id=principal.id,
    )
