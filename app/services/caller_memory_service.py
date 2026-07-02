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
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.core.logger import logger
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession

_CACHE_KEY = "caller_memory_context"
_DEFAULT_FETCH_TIMEOUT_SEC = 0.1


def _fetch_recent_summaries(
    db: Session, call_session: CallSession, window: int
) -> List[CallSession]:
    """Blocking DB query — run inside a threadpool executor with a timeout."""
    stmt = (
        select(CallSession)
        .where(
            CallSession.tenant_id == call_session.tenant_id,
            CallSession.call_flow_id == call_session.call_flow_id,
            CallSession.from_number == call_session.from_number,
            CallSession.status == "completed",
            CallSession.id != call_session.id,
            CallSession.transcript_summary.isnot(None),
            CallSession.transcript_summary != "",
        )
        .order_by(CallSession.start_time.desc())
        .limit(window)
    )
    return list(db.execute(stmt).scalars().all())


def _format_caller_memory_block(sessions: List[CallSession]) -> str:
    if not sessions:
        return ""

    lines = [f"CALLER HISTORY (last {len(sessions)} interactions):"]
    for session in sessions:
        date_str = (
            session.start_time.strftime("%Y-%m-%d") if session.start_time else "unknown date"
        )
        summary = (session.transcript_summary or "").strip()
        lines.append(f"- Call on {date_str}: {summary}")
    lines.append("End of caller history.")
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
    try:
        loop = asyncio.get_running_loop()
        sessions = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_recent_summaries, db, call_session, window),
            timeout=float(
                getattr(settings, "VOICE_CALLER_MEMORY_FETCH_TIMEOUT_SEC", None)
                or _DEFAULT_FETCH_TIMEOUT_SEC
            ),
        )
        context_block = _format_caller_memory_block(sessions)
    except asyncio.TimeoutError:
        logger.warning(
            "caller_memory lookup timed out after %sms; proceeding without caller memory",
            int(_DEFAULT_FETCH_TIMEOUT_SEC * 1000),
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
