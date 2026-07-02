"""
v2 A/B prompt testing endpoints.

Auth: any authenticated tenant principal (API key or JWT); config-rank required
to mutate A/B settings, read-only rank is sufficient to view results.

PUT  /api/v2/flows/{flow_id}/ab-test
GET  /api/v2/flows/{flow_id}/ab-results
PUT  /api/v2/flows/{flow_id}/ab-test/winner
"""
from __future__ import annotations

import uuid
from typing import Union

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_config_or_api_key, require_readonly_or_api_key
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User
from app.schemas.ab_testing import (
    AbResultsResponse,
    AbTestResponse,
    AbTestUpdate,
    AbTestWinnerUpdate,
)
from app.services.call_flow_service import call_flow_service

router = APIRouter(prefix="/flows", tags=["A/B Prompt Testing"])


def _tenant_id(principal: Union[User, ApiKeyPrincipal]) -> uuid.UUID:
    return principal.current_tenant_id


@router.put(
    "/{flow_id}/ab-test",
    response_model=AbTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Configure A/B prompt testing on a call flow",
)
def update_ab_test(
    flow_id: uuid.UUID,
    body: AbTestUpdate,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_config_or_api_key),
    db: Session = Depends(get_db),
) -> AbTestResponse:
    return call_flow_service.update_ab_test(db, flow_id, _tenant_id(principal), body)


@router.get(
    "/{flow_id}/ab-results",
    response_model=AbResultsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get A/B prompt test results and statistical significance",
)
def get_ab_results(
    flow_id: uuid.UUID,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
) -> AbResultsResponse:
    return call_flow_service.get_ab_results(db, flow_id, _tenant_id(principal))


@router.put(
    "/{flow_id}/ab-test/winner",
    status_code=status.HTTP_200_OK,
    summary="Promote the winning A/B variant to the flow's active prompt",
)
def promote_ab_winner(
    flow_id: uuid.UUID,
    body: AbTestWinnerUpdate,
    principal: Union[User, ApiKeyPrincipal] = Depends(require_config_or_api_key),
    db: Session = Depends(get_db),
) -> dict:
    return call_flow_service.promote_ab_winner(db, flow_id, _tenant_id(principal), body)
