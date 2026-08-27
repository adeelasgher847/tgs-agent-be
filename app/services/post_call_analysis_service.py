"""
Post-Call Analysis — tenant-defined custom variable extraction, configured
per call flow via ``PUT /api/v2/flows/{flow_id}/post-call-analysis-settings``.

Independent of, and additive to, the always-on hardcoded call-summary
pipeline in `voice_analysis_service.py` (summary/sentiment/caller_name/
success_evaluation/recommendations). Empty `post_call_analysis_variables` on
a `CallFlow` means this whole feature is a no-op — the hardcoded pipeline
keeps running unaffected either way.

Fail-open, fire-and-forget: this must never affect call completion, the base
call-summary job, or the Post Call Actions email send. Reuses the same
provider-dispatch helper (`generate_analysis_text`) as
`voice_analysis_service.py`, but resolves and persists a distinct
`call_metadata["post_call_analysis"]` block, and serializes on its own
advisory-lock key so it never unnecessarily contends with the hardcoded
summary's lock.
"""

from __future__ import annotations

import asyncio
import json
import re
import struct
import uuid
from datetime import datetime, timezone
from typing import Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.logger import logger
from app.core.security import decrypt_api_key
from app.db.session import SessionLocal
from app.models.call_flow import CallFlow
from app.models.call_log import CallLog
from app.models.call_session import CallSession
from app.services.agent_service import agent_service
from app.services.dlp_service import redact_phi_if_hipaa
from app.services.model_service import ModelService
from app.services.transcript_service import transcript_service
from app.services.voice_analysis_service import generate_analysis_text
from app.utils.arq_pool import get_arq_pool

# Fixed namespace UUID (arbitrary, stable) used to salt the call_session_id
# before deriving advisory-lock int32 keys, so this feature's lock key space
# never collides with voice_analysis_service._pg_advisory_key_pair's.
_LOCK_NAMESPACE = uuid.UUID("6f2a2f4e-6e9a-4b7a-9f9f-1c0e1b2a3d4f")

_model_service = ModelService()


def schedule_run_post_call_analysis(call_session_id: uuid.UUID) -> None:
    """
    Enqueue custom post-call variable extraction as an ARQ background job.
    Fire-and-forget — never blocks the caller. Fails open if the ARQ pool
    isn't ready: this is best-effort and must never affect call completion.
    """
    pool = get_arq_pool()
    if pool is None:
        logger.warning(
            "ARQ pool not ready; post-call analysis skipped for session=%s",
            call_session_id,
        )
        return

    async def _enqueue() -> None:
        try:
            await pool.enqueue_job("run_post_call_analysis", str(call_session_id))
        except Exception as exc:
            logger.warning("Failed to enqueue post-call analysis job: %s", exc)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_enqueue())
        return
    asyncio.create_task(_enqueue())


def _pg_advisory_key_pair(call_session_id: uuid.UUID) -> Tuple[int, int]:
    """Two int32 keys derived from call_session_id, salted into a distinct
    namespace so this feature's lock key space never collides with
    voice_analysis_service._pg_advisory_key_pair's (which derives directly
    from call_session_id's own bytes)."""
    salted = uuid.uuid5(_LOCK_NAMESPACE, str(call_session_id))
    return struct.unpack(">ii", salted.bytes[:8])


def _try_acquire_post_call_analysis_lock(db: Session, call_session_id: uuid.UUID) -> bool:
    """
    Serialize custom post-call analysis extraction for one call_session across
    workers. Returns False if lock unavailable (another worker is generating);
    caller should exit quietly. Non-blocking (`pg_try_advisory_lock`), no-op on
    non-postgres dialects (e.g. sqlite test DBs).
    """
    try:
        if db.get_bind().dialect.name != "postgresql":
            return True
        k1, k2 = _pg_advisory_key_pair(call_session_id)
        row = db.execute(
            text("SELECT pg_try_advisory_lock(:k1, :k2) AS ok"),
            {"k1": k1, "k2": k2},
        ).mappings().first()
        return bool(row and row.get("ok"))
    except Exception:
        logger.warning(
            "Post-call analysis: advisory lock check failed (continuing without lock)",
            exc_info=True,
        )
        return True


def _release_post_call_analysis_lock(db: Session, call_session_id: uuid.UUID) -> None:
    try:
        if db.get_bind().dialect.name != "postgresql":
            return
        k1, k2 = _pg_advisory_key_pair(call_session_id)
        db.execute(text("SELECT pg_advisory_unlock(:k1, :k2)"), {"k1": k1, "k2": k2})
        db.commit()
    except Exception:
        logger.warning("Post-call analysis: pg_advisory_unlock failed (non-critical)", exc_info=True)
    finally:
        try:
            db.rollback()
        except Exception:  # noqa: S110 - already logged the advisory-unlock failure above
            pass


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fence(content: str) -> str:
    stripped = content.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        return m.group(1).strip()
    return stripped


def _parse_extraction_response(content: str, variable_names: list[str]) -> dict[str, str]:
    """Parse the model's JSON response into {name: value}. Tolerates a fenced
    code block; falls back to a best-effort per-key regex extraction if the
    content isn't valid JSON (or doesn't parse to a dict)."""
    candidate = _strip_code_fence(content)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return {
                name: str(parsed[name])
                for name in variable_names
                if name in parsed
            }
    except json.JSONDecodeError:
        pass

    # Best-effort fallback: regex-extract each configured key individually.
    result: dict[str, str] = {}
    for name in variable_names:
        m = re.search(rf'"{re.escape(name)}"\s*:\s*"([^"]*)"', content)
        if m:
            result[name] = m.group(1)
    return result


def _run_extraction(db: Session, call_session: CallSession, call_flow: CallFlow) -> None:
    """Perform the LLM extraction pass and persist results. Caller
    (`ensure_post_call_analysis_locked`) must hold the advisory lock and have
    already re-checked the cache before calling this."""
    variables: list[dict] = list(call_flow.post_call_analysis_variables or [])
    if not variables:
        return

    transcript_messages = transcript_service.get_messages_by_session(db, call_session.id)
    if not transcript_messages:
        logger.info(
            "Post-call analysis skipped — no transcript messages for session %s",
            call_session.id,
        )
        return

    transcript_text = ""
    for msg in transcript_messages:
        role_label = "Agent" if msg.role == "agent" else "Customer"
        transcript_text += f"{role_label}: {msg.message}\n"

    variable_names = [v["name"] for v in variables]
    instructions = "\n".join(
        f'- "{v["name"]}" (type: {v.get("type", "string")}): {v["description"]}'
        for v in variables
    )

    prompt = f"""
Analyze this call transcript and extract the following variables.

Call Transcript:
{transcript_text}

Variables to extract (JSON key: extraction instruction):
{instructions}

Output ONLY a single raw JSON object whose keys are exactly the variable
names listed above and whose values are short strings answering each
instruction. If the transcript does not contain the requested information for
a variable, use exactly the string "not mentioned" as its value. Do not
include any prose, explanation, or markdown code fences — output raw JSON
only.
"""

    # Resolve agent's preferred model, same lookup as analyze_call_transcript.
    preferred_model: str | None = None
    if call_session.agent_id:
        try:
            agent = agent_service.get_agent_by_id(
                db, call_session.agent_id, call_session.tenant_id
            )
            if agent and agent.model:
                preferred_model = agent.model.model_name
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "Post-call analysis: could not resolve agent's preferred model: %s", e
            )

    candidates = [
        call_flow.post_call_analysis_model,
        preferred_model,
        "gemini-2.0-flash",
        "llama-3.3-70b-versatile",
        "gpt-4o-mini",
    ]
    seen: set[str] = set()
    fallback_models: list[str] = []
    for m in candidates:
        if m and m not in seen:
            seen.add(m)
            fallback_models.append(m)

    used_model: str | None = None
    content: str | None = None
    for model_name in fallback_models:
        try:
            model = _model_service.get_model_by_name(db, model_name)
            if not model:
                continue
            api_key = None
            if model.api_key:
                api_key = decrypt_api_key(model.api_key)
            result = generate_analysis_text(model, api_key, prompt, max_tokens=500)
            content = (result.get("content") or "").strip()
            used_model = model.model_name
            break
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "Post-call analysis: model %s failed, trying next: %s", model_name, e
            )
            continue

    if not content or not used_model:
        logger.warning(
            "Post-call analysis: all candidate models failed for session %s", call_session.id
        )
        return

    parsed = _parse_extraction_response(content, variable_names)
    if not parsed:
        logger.warning(
            "Post-call analysis: could not parse any variables from model response "
            "for session %s, leaving unpersisted for retry",
            call_session.id,
        )
        return

    # At least one variable parsed successfully. Backfill any configured
    # variable names the model/parser didn't produce a value for with the
    # same "not mentioned" sentinel the prompt already instructs the model
    # to use for genuinely-absent information, so a parse-failure and a
    # genuine "not mentioned" answer are indistinguishable to readers of the
    # persisted result, and `parsed` is always fully populated (one entry
    # per configured variable) before persistence — this keeps the cache
    # check in `ensure_post_call_analysis_locked` from ever treating a
    # partial extraction as a complete, non-retriable result.
    missing_names = [name for name in variable_names if name not in parsed]
    if missing_names:
        logger.warning(
            "Post-call analysis: model response for session %s did not include "
            "values for variables %s; backfilling with 'not mentioned'",
            call_session.id,
            missing_names,
        )
        for name in missing_names:
            parsed[name] = "not mentioned"

    hipaa_enabled = bool(call_flow.hipaa_compliance)
    redacted = {
        name: redact_phi_if_hipaa(value, hipaa_enabled=hipaa_enabled)
        for name, value in parsed.items()
    }

    analysis_block = {
        "variables": redacted,
        "model_used": used_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if call_session.call_metadata is None:
        call_session.call_metadata = {}
    call_session.call_metadata["post_call_analysis"] = analysis_block
    flag_modified(call_session, "call_metadata")
    db.add(call_session)

    log_row = db.query(CallLog).filter(CallLog.call_session_id == call_session.id).first()
    if log_row:
        if log_row.call_metadata is None:
            log_row.call_metadata = {}
        log_row.call_metadata["post_call_analysis"] = analysis_block
        flag_modified(log_row, "call_metadata")
        db.add(log_row)

    db.commit()
    logger.info(
        "Post-call analysis completed for session %s using %s", call_session.id, used_model
    )


def ensure_post_call_analysis_locked(
    db: Session, call_session: CallSession, call_flow: CallFlow
) -> None:
    """
    Double-checked-locking wrapper for lazily computing custom post-call
    variable extraction, matching `voice_analysis_service.ensure_call_summary_locked`'s
    shape but keyed on a distinct advisory-lock namespace. No-ops quietly if
    results are already cached, if the flow has no configured variables, or if
    another worker currently holds the lock (`pg_try_advisory_lock` is
    non-blocking).
    """
    if call_session.call_metadata and "post_call_analysis" in call_session.call_metadata:
        return
    if not (call_flow.post_call_analysis_variables or []):
        return

    session_uuid = call_session.id
    if not _try_acquire_post_call_analysis_lock(db, session_uuid):
        logger.debug(
            "Post-call analysis: another worker is already generating results for "
            "this call, skipping session=%s",
            session_uuid,
        )
        return

    try:
        db.refresh(call_session)
        if call_session.call_metadata and "post_call_analysis" in call_session.call_metadata:
            logger.debug(
                "Post-call analysis skipped — cached by another worker while waiting "
                "for lock, session %s",
                session_uuid,
            )
            return

        _run_extraction(db, call_session, call_flow)
    finally:
        _release_post_call_analysis_lock(db, session_uuid)


def _run_post_call_analysis_impl(call_session_id: uuid.UUID) -> None:
    """
    Real task body — invoked by the ARQ wrapper (`run_post_call_analysis` in
    `app/workers/batch_call_worker.py`). Wrapped entirely in a try/except so
    it can never raise into the ARQ worker in a way that breaks other jobs.
    """
    db: Session = SessionLocal()
    try:
        call_session = db.query(CallSession).filter(CallSession.id == call_session_id).first()
        if not call_session:
            logger.warning(
                "Post-call analysis skipped — session not found: %s", call_session_id
            )
            return

        call_flow = None
        if call_session.call_flow_id:
            call_flow = db.query(CallFlow).filter(CallFlow.id == call_session.call_flow_id).first()

        if not call_flow or not (call_flow.post_call_analysis_variables or []):
            logger.debug(
                "Post-call analysis skipped — no configured variables for session=%s",
                call_session_id,
            )
            return

        ensure_post_call_analysis_locked(db, call_session, call_flow)
    except Exception as exc:
        logger.warning(
            "Post-call analysis failed (non-critical) session=%s: %s",
            call_session_id,
            exc,
        )
    finally:
        db.close()
