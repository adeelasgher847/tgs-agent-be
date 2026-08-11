"""
Post-call "Email Summary" / "Summary to Business Owner" actions, configured
per call flow via ``PUT /api/v2/flows/{flow_id}/post-call-actions-settings``.

Fail-open, fire-and-forget: this must never affect call completion or race
with any of `call_session_service.py`'s other post-call hooks. Reuses the
LLM call analysis already computed (or lazily computed here) by
`voice_analysis_service.analyze_call_transcript`, so this is a no-op LLM-wise
if `schedule_call_summary_generation` already ran for this call.
"""

from __future__ import annotations

import asyncio
import html
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.db.session import SessionLocal
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.user import User, user_tenant_association
from app.services.email_service import email_service
from app.services.voice_analysis_service import ensure_call_summary_locked
from app.utils.arq_pool import get_arq_pool


def schedule_post_call_email_summary(call_session_id: uuid.UUID) -> None:
    """
    Enqueue the post-call email summary job as an ARQ background job.
    Fire-and-forget — never blocks the caller. Fails open if the ARQ pool
    isn't ready: this is best-effort and must never affect call completion.
    """
    pool = get_arq_pool()
    if pool is None:
        logger.warning(
            "ARQ pool not ready; post-call email summary skipped for session=%s",
            call_session_id,
        )
        return

    async def _enqueue() -> None:
        try:
            await pool.enqueue_job("send_post_call_email_summary", str(call_session_id))
        except Exception as exc:
            logger.warning("Failed to enqueue post-call email summary job: %s", exc)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_enqueue())
        return
    asyncio.create_task(_enqueue())


def _resolve_owner_email(db: Session, tenant_id: uuid.UUID) -> str | None:
    owner_user_id = db.execute(
        select(user_tenant_association.c.user_id).where(
            user_tenant_association.c.tenant_id == tenant_id,
            user_tenant_association.c.is_creator.is_(True),
        )
    ).scalar_one_or_none()
    if not owner_user_id:
        return None
    owner = db.query(User).filter(User.id == owner_user_id).first()
    return owner.email if owner and owner.email else None


def _build_email_content(
    call_session: CallSession, call_flow: CallFlow | None
) -> tuple[str, str]:
    analysis_data: dict = {}
    if call_session.call_metadata and "llm_call_analysis" in call_session.call_metadata:
        analysis_data = call_session.call_metadata["llm_call_analysis"].get("analysis") or {}

    caller_name = analysis_data.get("caller_name") or "Unknown caller"
    flow_name = call_flow.name if call_flow else "Call Flow"
    subject = f"Call Summary — {caller_name} — {flow_name}"

    summary_text = call_session.transcript_summary or analysis_data.get("summary") or "No summary available."
    sentiment = analysis_data.get("sentiment")
    success_evaluation = analysis_data.get("success_evaluation")
    recommendations = analysis_data.get("recommendations") or []

    recommendations_html = ""
    if recommendations:
        items = "".join(f"<li>{r}</li>" for r in recommendations)
        recommendations_html = f"<p><strong>Recommendations:</strong></p><ul>{items}</ul>"

    extracted_variables: dict = {}
    if call_session.call_metadata and "post_call_analysis" in call_session.call_metadata:
        extracted_variables = (
            call_session.call_metadata["post_call_analysis"].get("variables") or {}
        )

    extracted_details_html = ""
    if extracted_variables:
        items = "".join(
            f"<li><strong>{html.escape(name.replace('_', ' ').title())}:</strong> "
            f"{html.escape(str(value))}</li>"
            for name, value in extracted_variables.items()
        )
        extracted_details_html = (
            f"<p><strong>Extracted Details:</strong></p><ul>{items}</ul>"
        )

    html_body = f"""
    <html>
    <body>
        <h2>Call Summary</h2>
        <p><strong>Call Flow:</strong> {flow_name}</p>
        <p><strong>Caller:</strong> {caller_name}</p>
        <p><strong>Summary:</strong> {summary_text}</p>
        {f"<p><strong>Sentiment:</strong> {sentiment}</p>" if sentiment else ""}
        {f"<p><strong>Outcome:</strong> {success_evaluation}</p>" if success_evaluation else ""}
        {recommendations_html}
        {extracted_details_html}
        <p>Best regards,<br>The Voice Agent Team</p>
    </body>
    </html>
    """
    return subject, html_body


def _send_post_call_email_summary_impl(call_session_id: uuid.UUID) -> None:
    """
    Real task body — invoked by the ARQ wrapper (`send_post_call_email_summary`
    in `app/workers/batch_call_worker.py`). Wrapped entirely in a try/except so
    it can never raise into the ARQ worker in a way that breaks other jobs.
    """
    db: Session = SessionLocal()
    try:
        call_session = db.query(CallSession).filter(CallSession.id == call_session_id).first()
        if not call_session:
            logger.warning(
                "Post-call email summary skipped — session not found: %s", call_session_id
            )
            return

        call_flow = None
        if call_session.call_flow_id:
            call_flow = db.query(CallFlow).filter(CallFlow.id == call_session.call_flow_id).first()

        if not call_flow or not (
            call_flow.email_summary_enabled or call_flow.summary_to_business_owner_enabled
        ):
            logger.debug(
                "Post-call email summary skipped — neither toggle enabled for session=%s",
                call_session_id,
            )
            return

        # Match `analyze_call_transcript`'s own cache check (presence of the
        # "llm_call_analysis" key), not the truthiness of its nested "analysis"
        # sub-key — otherwise a partially-failed prior run (key present, empty
        # analysis) looks like a cache miss here but still short-circuits inside
        # analyze_call_transcript, leaving us with the same stale/empty block.
        has_analysis = bool(
            call_session.call_metadata
            and "llm_call_analysis" in call_session.call_metadata
        )
        if not has_analysis:
            try:
                # Serializes against the automatic `generate_call_summary` ARQ job
                # via the same pg advisory lock, so the two jobs never both run
                # the (paid) LLM analysis concurrently for the same call.
                ensure_call_summary_locked(
                    db,
                    call_session,
                    raise_on_no_transcript=False,
                )
            except Exception as analysis_exc:
                logger.warning(
                    "Post-call email summary: analysis generation failed (non-critical) "
                    "session=%s: %s",
                    call_session_id,
                    analysis_exc,
                )

        if call_flow.post_call_analysis_variables and not (
            call_session.call_metadata
            and "post_call_analysis" in call_session.call_metadata
        ):
            try:
                from app.services.post_call_analysis_service import (
                    ensure_post_call_analysis_locked,
                )

                # Best-effort — a slow/failed custom extraction must never
                # block the base email send.
                ensure_post_call_analysis_locked(db, call_session, call_flow)
            except Exception as analysis_exc:
                logger.warning(
                    "Post-call email summary: custom variable extraction failed "
                    "(non-critical) session=%s: %s",
                    call_session_id,
                    analysis_exc,
                )

        recipients: list[str] = []
        if call_flow.email_summary_enabled:
            recipients.extend(list(call_flow.email_summary_recipients or []))

        if call_flow.summary_to_business_owner_enabled:
            owner_email = _resolve_owner_email(db, call_session.tenant_id)
            if owner_email and owner_email not in recipients:
                recipients.append(owner_email)

        if not recipients:
            logger.info(
                "Post-call email summary: no recipients resolved for session=%s, skipping",
                call_session_id,
            )
            return

        subject, html_body = _build_email_content(call_session, call_flow)
        email_service.send_generic_email(
            to_email=recipients[0],
            subject=subject,
            html_body=html_body,
            cc_emails=recipients[1:] if len(recipients) > 1 else None,
        )
    except Exception as exc:
        logger.warning(
            "Post-call email summary send failed (non-critical) session=%s: %s",
            call_session_id,
            exc,
        )
    finally:
        db.close()
