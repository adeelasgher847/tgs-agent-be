"""Web SDK domain whitelist management — /api/v1/workspace/allowed-domains.

Registered under the ``/workspace`` prefix in app/api/api_v1/api.py, and
must be included BEFORE ``workspace.router`` so this literal path takes
priority over that router's ``/{workspace_id}`` catch-all — same reasoning
as workspace_invites.router.
"""
from __future__ import annotations

import uuid
from typing import Union

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User
from app.schemas.allowed_domain import AllowedDomainCreate, AllowedDomainOut
from app.schemas.base import SuccessResponse
from app.services.allowed_domain_service import allowed_domain_service
from app.utils.response import create_success_response

router = APIRouter()


def _workspace_id(principal: Union[User, ApiKeyPrincipal]) -> uuid.UUID:
    return principal.current_tenant_id


@router.post(
    "/allowed-domains",
    response_model=AllowedDomainOut,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
    summary="Whitelist a domain for the Web SDK",
)
def create_allowed_domain(
    body: AllowedDomainCreate,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    return allowed_domain_service.create_domain(db, _workspace_id(principal), body)


@router.get(
    "/allowed-domains",
    response_model=SuccessResponse[list[AllowedDomainOut]],
    response_model_by_alias=True,
    summary="List domains whitelisted for the Web SDK",
)
def list_allowed_domains(
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    domains = allowed_domain_service.list_domains(db, _workspace_id(principal))
    return create_success_response(domains, "Allowed domains retrieved successfully")


@router.delete(
    "/allowed-domains/{domain_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Remove a whitelisted domain",
)
def delete_allowed_domain(
    domain_id: uuid.UUID,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    allowed_domain_service.delete_domain(db, _workspace_id(principal), domain_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
