"""
Push inbound call logs to tenant CRM (Trello). Does not touch scheduled-calls CRM.

When inbound CRM is enabled, each completed inbound call runs the same transcript analysis
as /voice/transcript/analyze (LLM), then pushes summary, sentiment, recommendations,
caller name, and transcript-based success evaluation to Trello — not the DB success field.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.core.config import settings
from app.core.logger import logger
from app.db.session import SessionLocal
from app.models.call_log import CallLog
from app.models.call_log_crm_sync import CallLogCRMSync
from app.models.call_session import CallSession
from app.models.tenant_inbound_crm_config import TenantInboundCRMConfig
from app.services.trello_service import TrelloService


def _extract_llm_analysis(call_log: CallLog, session: CallSession) -> Optional[Dict[str, Any]]:
    for meta in (call_log.call_metadata, session.call_metadata):
        if not isinstance(meta, dict):
            continue
        block = meta.get("llm_call_analysis")
        if isinstance(block, dict):
            inner = block.get("analysis")
            if isinstance(inner, dict) and inner:
                return inner
    return None


def _build_card_description(call_log: CallLog, session: CallSession) -> str:
    phone = call_log.customer_phone_number or session.customer_phone_number or "Unknown"
    analysis = _extract_llm_analysis(call_log, session)

    if analysis and analysis.get("success_evaluation"):
        success_eval = str(analysis["success_evaluation"]).strip()
    else:
        success_eval = (
            "_Not available — no transcript or analysis did not complete._"
        )

    caller_name = "—"
    if analysis and analysis.get("caller_name"):
        caller_name = str(analysis["caller_name"]).strip() or "—"

    parts = [
        "## Customer",
        f"- **Phone**: {phone}",
        f"- **Name** (from analysis): {caller_name}",
        "",
        "## Success evaluation",
        f"{success_eval}",
        "",
        "## Sentiment",
    ]
    if analysis and analysis.get("sentiment"):
        parts.append(str(analysis["sentiment"]).strip())
    else:
        parts.append("_Not available — analysis did not run or transcript was empty._")

    parts += ["", "## Analysis report"]
    if analysis and analysis.get("summary"):
        parts.append(str(analysis["summary"]).strip())
    else:
        parts.append("_Not available._")

    recs = analysis.get("recommendations") if analysis else None
    if isinstance(recs, list) and recs:
        parts += ["", "### Recommendations"]
        for i, rec in enumerate(recs, 1):
            parts.append(f"{i}. {rec}")

    parts += [
        "",
        "## Call reference",
        f"- Session: `{session.id}`",
        f"- Log: `{call_log.id}`",
        f"- Duration (s): {call_log.duration if call_log.duration is not None else session.duration}",
    ]
    return "\n".join(parts)


def _card_title(call_log: CallLog, session: CallSession) -> str:
    analysis = _extract_llm_analysis(call_log, session)
    cn = (analysis or {}).get("caller_name") if analysis else None
    if cn and str(cn).strip() and str(cn).strip().lower() != "unknown":
        caller_display = str(cn).strip()[:80]
    else:
        caller_display = call_log.customer_phone_number or session.customer_phone_number or "Unknown"
    return f"Inbound | {caller_display} | {str(call_log.id)[:8]}"


def _trello_for_config(config: TenantInboundCRMConfig) -> TrelloService:
    if config.connection_type == "platform_managed":
        k = settings.TRELLO_PLATFORM_API_KEY or ""
        t = settings.TRELLO_PLATFORM_API_TOKEN or ""
        if not k or not t:
            raise ValueError("Platform Trello is not configured (TRELLO_PLATFORM_API_KEY / TOKEN).")
        return TrelloService(api_key=k, api_token=t)
    key = config.encrypted_api_key or ""
    token = config.encrypted_api_token or ""
    if not key or not token:
        raise ValueError("Missing Trello API key or token for this tenant.")
    return TrelloService(api_key=key, api_token=token)


def tenant_has_active_inbound_crm(db: Session, tenant_id: uuid.UUID) -> bool:
    """True when tenant has inbound Trello CRM enabled with a board — mirrors sync preconditions."""
    cfg = (
        db.query(TenantInboundCRMConfig)
        .filter(
            TenantInboundCRMConfig.tenant_id == tenant_id,
            TenantInboundCRMConfig.is_enabled.is_(True),
            TenantInboundCRMConfig.provider == "trello",
        )
        .first()
    )
    return bool(cfg and cfg.container_id and str(cfg.container_id).strip())


def sync_inbound_call_to_crm(call_session_id: uuid.UUID) -> None:
    db: Session = SessionLocal()
    try:
        session = db.query(CallSession).filter(CallSession.id == call_session_id).first()
        if not session or (session.call_type or "").lower() != "inbound":
            return

        if (session.status or "").lower() not in ("completed", "failed", "busy"):
            return

        call_log = db.query(CallLog).filter(CallLog.call_session_id == session.id).first()
        if not call_log:
            logger.warning("Inbound CRM sync: no CallLog for session %s", call_session_id)
            return

        config = (
            db.query(TenantInboundCRMConfig)
            .filter(
                TenantInboundCRMConfig.tenant_id == session.tenant_id,
                TenantInboundCRMConfig.is_enabled.is_(True),
                TenantInboundCRMConfig.provider == "trello",
            )
            .first()
        )
        if not config:
            return

        if not config.container_id:
            logger.warning("Inbound CRM sync: no board (container_id) for tenant %s", session.tenant_id)
            return

        from app.services.voice_analysis_service import voice_analysis_service

        try:
            voice_analysis_service.analyze_call_transcript(
                db,
                session,
                session.user_id,
                raise_on_no_transcript=False,
            )
        except HTTPException as he:
            logger.warning(
                "Inbound CRM auto-analysis skipped or failed (HTTP): %s",
                he.detail,
            )
        except Exception:
            logger.exception("Inbound CRM auto-analysis failed (continuing with Trello sync)")

        session = db.query(CallSession).filter(CallSession.id == call_session_id).first()
        call_log = db.query(CallLog).filter(CallLog.call_session_id == session.id).first()
        if not session or not call_log:
            logger.warning("Inbound CRM sync: session/log missing after analysis")
            return

        sync_row = db.query(CallLogCRMSync).filter(CallLogCRMSync.call_log_id == call_log.id).first()
        if not sync_row:
            sync_row = CallLogCRMSync(
                call_log_id=call_log.id,
                tenant_inbound_crm_config_id=config.id,
                sync_status="pending",
                attempt_count=0,
            )
            db.add(sync_row)
            db.commit()
            db.refresh(sync_row)

        try:
            trello = _trello_for_config(config)
            list_id = config.default_list_id
            if not list_id:
                list_id = trello.ensure_inbound_call_logs_list(config.container_id)
                config.default_list_id = list_id
                db.add(config)
                db.commit()

            desc = _build_card_description(call_log, session)
            title = _card_title(call_log, session)

            if sync_row.external_item_id:
                result = trello.update_inbound_call_log_card(
                    sync_row.external_item_id,
                    card_name=title,
                    description=desc,
                )
            else:
                result = trello.create_inbound_call_log_card(list_id, title, desc)

            sync_row.external_item_id = result.get("id", sync_row.external_item_id)
            sync_row.external_item_url = result.get("url") or sync_row.external_item_url
            sync_row.sync_status = "success"
            sync_row.last_error = None
            sync_row.updated_at = datetime.now(timezone.utc)
            db.add(sync_row)
            db.commit()
            logger.info("Inbound CRM sync OK call_log=%s card=%s", call_log.id, sync_row.external_item_id)
        except Exception as e:
            logger.exception("Inbound CRM sync failed for session %s", call_session_id)
            sync_row = db.query(CallLogCRMSync).filter(CallLogCRMSync.call_log_id == call_log.id).first()
            if sync_row:
                sync_row.attempt_count = (sync_row.attempt_count or 0) + 1
                sync_row.sync_status = "failed"
                sync_row.last_error = str(e)[:2000]
                sync_row.updated_at = datetime.now(timezone.utc)
                db.add(sync_row)
                db.commit()
    finally:
        db.close()


async def sync_inbound_call_to_crm_async(call_session_id: uuid.UUID) -> None:
    await asyncio.to_thread(sync_inbound_call_to_crm, call_session_id)


def schedule_inbound_crm_sync(call_session_id: uuid.UUID) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        sync_inbound_call_to_crm(call_session_id)
        return
    asyncio.create_task(sync_inbound_call_to_crm_async(call_session_id))


def delete_tenant_inbound_crm_config(db: Session, tenant_id: uuid.UUID) -> Optional[dict]:
    """
    Remove this tenant's inbound CRM integration: delete only tracked Trello cards (not the board),
    then remove sync rows and config row. The Trello board itself is always left intact.
    """
    row = (
        db.query(TenantInboundCRMConfig)
        .filter(TenantInboundCRMConfig.tenant_id == tenant_id)
        .first()
    )
    if not row:
        return None

    trello_cards_deleted = 0

    if row.provider == "trello" and row.container_id:
        try:
            trello = _trello_for_config(row)
            syncs = (
                db.query(CallLogCRMSync)
                .filter(CallLogCRMSync.tenant_inbound_crm_config_id == row.id)
                .all()
            )
            for s in syncs:
                if s.external_item_id and trello.delete_card(s.external_item_id):
                    trello_cards_deleted += 1
        except Exception as e:
            logger.warning(
                "Trello card cleanup during inbound CRM delete failed (continuing with DB delete): %s",
                e,
            )

    cfg_id = row.id
    db.query(CallLogCRMSync).filter(
        CallLogCRMSync.tenant_inbound_crm_config_id == cfg_id
    ).delete(synchronize_session=False)
    db.delete(row)
    db.commit()

    return {
        "deleted": True,
        "trello_cards_deleted": trello_cards_deleted,
    }
