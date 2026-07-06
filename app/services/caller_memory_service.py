"""
Cross-session caller memory: injects summaries of a caller's previous completed
calls into the LLM system prompt so agents recall prior interactions.

Retrieval is fetched once per call and cached on call_session.call_metadata
(mirrors get_crm_context_block_for_call in hubspot_service.py) so later turns
in the same call — the prompt is rebuilt every turn — reuse the cached string
instead of re-querying the database. Fails open on timeout/error: a slow or
broken lookup never blocks the call, it just proceeds without caller memory.
"""
from __future__ import annotations

import asyncio
import datetime
import re
import uuid
from typing import List, NamedTuple, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.core.logger import logger
from app.db.session import SessionLocal
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession

_CACHE_KEY = "caller_memory_context"
_DEFAULT_FETCH_TIMEOUT_SEC = 0.1
_CONTROL_CHARS_RE = re.compile(r'[\x00-\x1f\x7f]')
_MAX_SUMMARY_LEN = 400


class CallerMemorySession(NamedTuple):
    start_time: Optional[datetime.datetime]
    transcript_summary: Optional[str]


def _sanitize_summary(text: str) -> str:
    text = _CONTROL_CHARS_RE.sub(' ', text).strip()
    if len(text) > _MAX_SUMMARY_LEN:
        text = text[:_MAX_SUMMARY_LEN] + '…'
    return text


def _fetch_recent_summaries(
    tenant_id: uuid.UUID,
    call_flow_id: uuid.UUID,
    from_number: str,
    current_session_id: uuid.UUID,
    window: int,
) -> List[CallerMemorySession]:
    """Blocking DB query — runs in a thread pool with its own session."""
    db = SessionLocal()
    try:
        stmt = (
            select(CallSession.start_time, CallSession.transcript_summary)
            .where(
                CallSession.tenant_id == tenant_id,
                CallSession.call_flow_id == call_flow_id,
                CallSession.from_number == from_number,
                CallSession.status == "completed",
                CallSession.id != current_session_id,
                CallSession.transcript_summary.isnot(None),
                CallSession.transcript_summary != "",
            )
            .order_by(CallSession.start_time.desc())
            .limit(window)
        )
        rows = db.execute(stmt).all()
        return [
            CallerMemorySession(start_time=row.start_time, transcript_summary=row.transcript_summary)
            for row in rows
        ]
    finally:
        db.close()


def _format_caller_memory_block(sessions: List[CallerMemorySession]) -> str:
    if not sessions:
        return ""

    lines = ["<caller_history>", f"CALLER HISTORY (last {len(sessions)} interactions):"]
    for session in sessions:
        date_str = (
            session.start_time.strftime("%Y-%m-%d") if session.start_time else "unknown date"
        )
        summary = _sanitize_summary((session.transcript_summary or "").strip())
        lines.append(f"- Call on {date_str}: {summary}")
    lines.append("End of caller history.")
    lines.append("</caller_history>")
    return "\n".join(lines)


async def get_caller_memory_context_block_for_call(
    db: Session,
    call_session: Optional[CallSession],
    call_flow: Optional[CallFlow],
) -> str:
    """
    Fetch the caller-memory context block for a call's system prompt, once per call.

    Returns "" (no block injected) when: caller memory is disabled on the flow,
    the call has no from_number, the lookup times out, or the lookup fails.
    """
    if call_session is None or call_flow is None or not call_flow.caller_memory_enabled:
        return ""
    if not call_session.from_number:
        return ""

    metadata = call_session.call_metadata or {}
    if _CACHE_KEY in metadata:
        return metadata.get(_CACHE_KEY) or ""

    window = call_flow.caller_memory_window
    context_block = ""
    fetch_timeout = float(
        getattr(settings, "VOICE_CALLER_MEMORY_FETCH_TIMEOUT_SEC", None)
        or _DEFAULT_FETCH_TIMEOUT_SEC
    )
    try:
        loop = asyncio.get_running_loop()
        sessions = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                _fetch_recent_summaries,
                call_session.tenant_id,
                call_session.call_flow_id,
                call_session.from_number,
                call_session.id,
                window,
            ),
            timeout=fetch_timeout,
        )
        context_block = _format_caller_memory_block(sessions)
    except asyncio.TimeoutError:
        logger.warning(
            "caller_memory lookup timed out after %dms; proceeding without caller memory",
            int(fetch_timeout * 1000),
        )
    except Exception:
        logger.warning(
            "caller_memory lookup failed; proceeding without caller memory", exc_info=True
        )

    try:
        # Nested savepoint keeps this fail-open and avoids committing the
        # caller's main transaction during prompt generation.
        with db.begin_nested():
            updated_metadata = dict(call_session.call_metadata or {})
            updated_metadata[_CACHE_KEY] = context_block
            call_session.call_metadata = updated_metadata
            flag_modified(call_session, "call_metadata")
            db.add(call_session)
            db.flush()
    except Exception:
        logger.warning(
            "Failed to cache caller memory context on call_session (non-critical)",
            exc_info=True,
        )

    return context_block
