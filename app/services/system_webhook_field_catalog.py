"""Field catalog for the Post-Call Webhook (System Webhooks / Call Flow).

Assembles a namespaced dict of everything a tenant can reference via
`{{namespace.field}}` tokens in a custom payload template
(`CallFlow.post_call_webhook_custom_payload_template`), and is also the
source for the default (non-custom) payload's `data` object.

No new storage — purely derived/additive from `CallSession`, `CallLog`,
`transcript_service`, and whatever `voice_analysis_service`/
`post_call_analysis_service` have already cached on
`CallSession.call_metadata` by the time the Post-Call Webhook fires.

Honesty about field provenance (read the source models/services before
trusting a field name — don't add to this catalog by guessing):
- `call_metadata` namespace: every field is a real, always-present `CallSession`
  column (nullable columns may still resolve to `None`).
- `conversation_data`, `analytics`, `header_variables` namespaces: best-effort.
  They read from `CallSession.call_metadata` JSONB blocks written by other
  background jobs (`post_call_analysis_service._run_extraction`,
  `voice_analysis_service.analyze_call_transcript`) that may not have run yet
  when the Post-Call Webhook fires, and — for `header_variables` — a
  `webhook_variables` key that only exists once the Pre-Inbound Call Webhook
  voice-pipeline wiring lands (not yet, as of this function's authorship).
  Treat every key under these three namespaces as possibly absent.
- `transcript` namespace: `full_transcript`/`message_count` are always computed
  fresh from `TranscriptMessage` rows (empty string / 0 if none exist yet).
  `transcript_url` is a hardcoded `None` placeholder — no such concept exists
  on `CallSession`/`CallLog` today (`recording_s3_path` is a storage key, not
  a fetchable URL).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models.call_log import CallLog
from app.models.call_session import CallSession
from app.services.transcript_service import transcript_service

_JsonPrimitive = str | int | float | bool | None


def _safe(value: Any) -> _JsonPrimitive:
    """Coerce an arbitrary value to a JSON-safe primitive for the payload."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_post_call_payload_context(
    db: Session,
    call_session: CallSession,
    call_log: CallLog | None,
) -> dict[str, dict[str, Any]]:
    """Return the field catalog for one completed call, namespaced for
    `{{namespace.field}}` template lookups (see `render_json_template`).

    `call_log` is accepted for parity with the plan's signature and future
    use, but every field currently cataloged is already available directly
    on `call_session` — `CallLog` largely duplicates the same columns for a
    separate logging table, not additive today.
    """
    call_metadata_ns: dict[str, Any] = {
        "call_id": str(call_session.id),
        "agent_id": _safe(call_session.agent_id),
        "tenant_id": _safe(call_session.tenant_id),
        "call_flow_id": _safe(call_session.call_flow_id),
        "call_type": call_session.call_type,
        "from_number": call_session.from_number,
        "to_number": call_session.to_number,
        "status": call_session.status,
        "duration": call_session.duration,
        "started_at": (
            call_session.start_time.isoformat() if call_session.start_time else None
        ),
        "ended_at": (
            call_session.end_time.isoformat() if call_session.end_time else None
        ),
        "ended_reason": call_session.ended_reason,
        "success_evaluation": call_session.success_evaluation,
        "cost": call_session.cost,
        "transferred": call_session.transferred,
        "twilio_call_sid": call_session.twilio_call_sid,
    }

    metadata = call_session.call_metadata or {}

    # conversation_data — flattened tenant-defined post-call-analysis
    # variables (post_call_analysis_service), plus caller_name cached by
    # voice_analysis_service.analyze_call_transcript. Best-effort: absent
    # until the respective background job has completed for this call.
    conversation_data_ns: dict[str, Any] = {}
    post_call_analysis = metadata.get("post_call_analysis") or {}
    for name, value in (post_call_analysis.get("variables") or {}).items():
        conversation_data_ns[name] = _safe(value)

    llm_analysis = (metadata.get("llm_call_analysis") or {}).get("analysis") or {}
    if llm_analysis.get("caller_name") is not None:
        conversation_data_ns["caller_name"] = _safe(llm_analysis.get("caller_name"))

    # transcript — role-labeled join, mirrors post_call_analysis_service's
    # `_run_extraction` transcript_text construction.
    transcript_messages = transcript_service.get_messages_by_session(
        db, call_session.id
    )
    full_transcript = ""
    for msg in transcript_messages:
        role_label = "Agent" if msg.role == "agent" else "Customer"
        full_transcript += f"{role_label}: {msg.message}\n"

    transcript_ns: dict[str, Any] = {
        "full_transcript": full_transcript,
        "message_count": len(transcript_messages),
        "transcript_url": None,  # placeholder — no such field exists yet
    }

    # analytics — same voice_analysis_service cached block used above for
    # caller_name. Best-effort/absent until analyze_call_transcript has run.
    recommendations = llm_analysis.get("recommendations")
    analytics_ns: dict[str, Any] = {
        "summary": _safe(llm_analysis.get("summary")),
        "recommendations": (
            list(recommendations) if isinstance(recommendations, list) else None
        ),
    }

    # header_variables — raw pass-through of whatever the Pre-Inbound Call
    # Webhook returned for this call. Not yet populated by any caller as of
    # this function's authorship (voice-pipeline agent's wiring is separate,
    # coming next) — always resolves to {} until that lands.
    header_variables_ns: dict[str, Any] = {
        k: _safe(v) for k, v in (metadata.get("webhook_variables") or {}).items()
    }

    return {
        "call_metadata": call_metadata_ns,
        "conversation_data": conversation_data_ns,
        "transcript": transcript_ns,
        "analytics": analytics_ns,
        "header_variables": header_variables_ns,
    }
