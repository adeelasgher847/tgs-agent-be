"""
Tenant inbound call log → CRM (Trello). Owner configures; all members benefit.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_owner, require_owner, require_tenant
from app.core.logger import logger
from app.core.security import encrypt_api_key
from app.models.tenant import Tenant
from app.models.user import User
from app.models.tenant_inbound_crm_config import TenantInboundCRMConfig
from app.schemas.base import SuccessResponse
from app.schemas.inbound_crm import (
    TenantInboundCRMConfigPublic,
    TenantInboundCRMConfigUpsert,
    TenantInboundCRMProvisionResponse,
    InboundBoardUrlOut,
)
from app.services.inbound_call_crm_sync_service import (
    _trello_for_config,
    delete_tenant_inbound_crm_config,
)
from app.services.trello_service import TrelloService
from app.utils.response import create_success_response

router = APIRouter()


def _to_public(row: TenantInboundCRMConfig) -> TenantInboundCRMConfigPublic:
    has_creds = bool(
        row.connection_type == "byo_credentials" and row.encrypted_api_key and row.encrypted_api_token
    )
    return TenantInboundCRMConfigPublic(
        id=row.id,
        tenant_id=row.tenant_id,
        provider=row.provider,
        connection_type=row.connection_type,
        container_id=row.container_id,
        container_url=row.container_url,
        is_enabled=row.is_enabled,
        has_credentials=has_creds or row.connection_type == "platform_managed",
    )


@router.get("/config", response_model=SuccessResponse[TenantInboundCRMConfigPublic | None])
def get_inbound_crm_config(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    row = (
        db.query(TenantInboundCRMConfig)
        .filter(TenantInboundCRMConfig.tenant_id == user.current_tenant_id)
        .first()
    )
    if not row:
        return create_success_response(None, "No inbound CRM configuration for this tenant")
    return create_success_response(_to_public(row), "Inbound CRM configuration")


@router.get("/board-url", response_model=SuccessResponse[InboundBoardUrlOut])
def get_inbound_crm_board_url(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Board link for this tenant to open Trello and view call cards (any tenant member)."""
    row = (
        db.query(TenantInboundCRMConfig)
        .filter(TenantInboundCRMConfig.tenant_id == user.current_tenant_id)
        .first()
    )
    if not row or not row.container_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No inbound CRM board configured for this tenant",
        )

    board_url = (row.container_url or "").strip()
    if not board_url:
        try:
            board_url = _trello_for_config(row).get_board_url(row.container_id)
            row.container_url = board_url
            db.add(row)
            db.commit()
        except Exception as e:
            logger.warning("Could not resolve Trello board URL from API: %s", e)
            board_url = f"https://trello.com/b/{row.container_id}"

    return create_success_response(
        InboundBoardUrlOut(board_url=board_url, board_id=row.container_id),
        "Inbound CRM board URL",
    )


@router.put("/config", response_model=SuccessResponse[TenantInboundCRMConfigPublic])
def upsert_inbound_crm_config(
    body: TenantInboundCRMConfigUpsert,
    user: User = Depends(require_owner),
    db: Session = Depends(get_db),
):
    if body.provider != "trello":
        raise HTTPException(status_code=400, detail="Only provider 'trello' is supported currently")

    tid = user.current_tenant_id
    if not tid:
        raise HTTPException(status_code=400, detail="No tenant selected")

    row = db.query(TenantInboundCRMConfig).filter(TenantInboundCRMConfig.tenant_id == tid).first()
    if not row:
        row = TenantInboundCRMConfig(
            tenant_id=tid,
            created_by_user_id=user.id,
        )
        db.add(row)

    row.provider = body.provider
    row.connection_type = body.connection_type
    row.is_enabled = body.is_enabled
    if body.extra_config is not None:
        row.extra_config = body.extra_config

    if body.connection_type == "byo_credentials":
        if body.api_key:
            row.encrypted_api_key = encrypt_api_key(body.api_key)
        if body.api_token:
            row.encrypted_api_token = encrypt_api_key(body.api_token)
    elif body.connection_type == "platform_managed":
        row.encrypted_api_key = None
        row.encrypted_api_token = None
    else:
        raise HTTPException(status_code=400, detail="Invalid connection_type")

    if body.board_url is not None:
        s = (body.board_url or "").strip()
        row.container_id = TrelloService.parse_board_id_from_url_or_id(s) if s else None
        if not row.container_id:
            row.default_list_id = None
    elif body.container_id is not None:
        s = (body.container_id or "").strip()
        row.container_id = TrelloService.parse_board_id_from_url_or_id(s) if s else None
        if not row.container_id:
            row.default_list_id = None

    if row.is_enabled:
        try:
            svc = _trello_for_config(row)
            if row.container_id:
                row.container_url = svc.get_board_url(row.container_id)
                if not row.default_list_id:
                    row.default_list_id = svc.ensure_inbound_call_logs_list(row.container_id)
            else:
                tenant = db.query(Tenant).filter(Tenant.id == tid).first()
                board_name = f"Call logs — {tenant.name if tenant else tid}"
                created = svc.create_container(board_name)
                board_id = created.get("id", "")
                if not board_id:
                    raise ValueError("Trello did not return a board id")
                row.container_id = board_id
                row.container_url = svc.get_board_url(board_id)
                row.default_list_id = svc.ensure_inbound_call_logs_list(board_id)
        except Exception as e:
            logger.warning("Could not verify/provision Trello board/list: %s", e)

    db.commit()
    db.refresh(row)
    return create_success_response(_to_public(row), "Inbound CRM configuration saved")


@router.post("/config/validate", response_model=SuccessResponse[dict])
def validate_inbound_crm_credentials(
    body: TenantInboundCRMConfigUpsert,
    user: User = Depends(require_owner),
    db: Session = Depends(get_db),
):
    if body.provider != "trello":
        raise HTTPException(status_code=400, detail="Only trello is supported")
    if not body.api_key or not body.api_token:
        raise HTTPException(status_code=400, detail="api_key and api_token required for validation")
    svc = TrelloService(api_key=body.api_key, api_token=body.api_token)
    try:
        info = svc.validate_credentials()
        return create_success_response(info, "Trello credentials are valid")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/config/provision-board", response_model=SuccessResponse[TenantInboundCRMProvisionResponse])
def provision_inbound_crm_board(
    user: User = Depends(require_owner),
    db: Session = Depends(get_db),
):
    tid = user.current_tenant_id
    row = db.query(TenantInboundCRMConfig).filter(TenantInboundCRMConfig.tenant_id == tid).first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Save inbound CRM config (keys / connection type) first",
        )

    tenant = db.query(Tenant).filter(Tenant.id == tid).first()
    name = f"Call logs — {tenant.name if tenant else tid}"

    try:
        svc = _trello_for_config(row)
        created = svc.create_container(name)
        board_id = created.get("id", "")
        if not board_id:
            raise HTTPException(status_code=502, detail="Trello did not return a board id")
        row.container_id = board_id
        row.container_url = svc.get_board_url(board_id)
        row.default_list_id = svc.ensure_inbound_call_logs_list(board_id)
        db.add(row)
        db.commit()
        return create_success_response(
            TenantInboundCRMProvisionResponse(
                board_id=board_id,
                board_url=row.container_url or "",
                list_id=row.default_list_id or "",
            ),
            "Board provisioned",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("provision_inbound_crm_board failed")
        raise HTTPException(status_code=502, detail=str(e))


@router.delete("/config", response_model=SuccessResponse[dict])
def delete_inbound_crm_config(
    user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    """
    Owner or admin: remove this tenant's cards from their Trello board (tracked in sync),
    then remove DB config. The Trello board itself is never deleted.
    """
    tid = user.current_tenant_id
    if not tid:
        raise HTTPException(status_code=400, detail="No tenant selected")

    result = delete_tenant_inbound_crm_config(db, tid)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No inbound CRM configuration found for this tenant",
        )

    return create_success_response(
        result,
        "Inbound CRM disconnected; call cards removed from Trello where possible; board kept on Trello",
    )
