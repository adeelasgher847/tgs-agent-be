import asyncio
import re
import struct
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from fastapi import HTTPException

from app.core.logger import logger
from app.core.pii_redactor import redact_pii
from app.db.session import SessionLocal
from app.models.call_flow import CallFlow
from app.models.call_log import CallLog
from app.models.call_session import CallSession
from app.services.agent_service import agent_service
from app.services.dlp_service import redact_phi_if_hipaa
from app.services.model_service import ModelService
from app.services.transcript_service import transcript_service
from app.utils.arq_pool import get_arq_pool


def generate_analysis_text(current_model, current_api_key, prompt: str, max_tokens: int = 200):
    """Generate text using the appropriate service based on provider.

    Lifted out of `analyze_call_transcript`'s private closure to module scope
    so other extraction pipelines (e.g. post_call_analysis_service.py) can
    reuse the same provider dispatch without duplicating it. Behavior is
    unchanged — same signature, same branching, same calls.
    """
    provider_name = (current_model.provider.name or "").strip().lower()

    if provider_name in (
        "gemini",
        "google",
        "google-ai",
        "google ai",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
    ):
        from app.services.gemini_service import GeminiService

        service = GeminiService()
        return service.generate_text(
            prompt=prompt,
            model_name=current_model.model_name,
            temperature=0.3,
            max_tokens=max_tokens,
            api_key=current_api_key,
        )
    elif provider_name in ("openai", "gpt", "gpt-4o-mini", "gpt-4o", "gpt-4"):
        from app.services.openai_service import OpenAIService

        service = OpenAIService()
        return service.generate_text(
            prompt=prompt,
            system_prompt="You are an AI assistant that analyzes call transcripts.",
            model_name=current_model.model_name,
            temperature=0.3,
            max_tokens=max_tokens,
            api_key=current_api_key,
        )
    elif provider_name in ("groq", "llama", "llama-3.3-70b-versatile"):
        from app.services.groq_service import GroqService

        service = GroqService()
        return service.generate_text(
            prompt=prompt,
            system_prompt="You are an AI assistant that analyzes call transcripts.",
            model_name=current_model.model_name,
            temperature=0.3,
            max_tokens=max_tokens,
            api_key=current_api_key,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider for analysis: {provider_name}",
        )


class VoiceAnalysisService:
    """Service responsible for transcript-based call analysis."""

    def __init__(self) -> None:
        self.model_service = ModelService()

    def analyze_call_transcript(
        self,
        db: Session,
        call_session: CallSession,
        user_id,
        raise_on_no_transcript: bool = True,
    ) -> Dict[str, Any] | None:
        """Behavior-preserving refactor of `analyze_call_transcript` logic from `voice.py`."""

        # Check if user has access to this call session
        # (same logic as original route)
        # NOTE: caller is responsible for validating call_session_id format and existence.

        # We don't have full User here, just ensure tenant/user match is enforced
        # by caller before invoking this method when necessary.

        # 🎯 FLEXIBLE MODEL SELECTION WITH FALLBACK
        # Priority: 1. Call's model, 2. Gemini 2.0 Flash, 3. Llama, 4. GPT-4o Mini

        # 🚀 CACHING: Return existing analysis if already present
        if call_session.call_metadata and "llm_call_analysis" in call_session.call_metadata:
            cached_block = call_session.call_metadata["llm_call_analysis"]
            
            # Reconstruct transcript message count for the response
            transcript_messages = transcript_service.get_messages_by_session(
                db, call_session.id
            )
            
            logger.info("📦 Transcript analysis fetched from DB for session %s", call_session.id)
            return {
                "call_session_id": str(call_session.id),
                "transcript_message_count": len(transcript_messages),
                "call_duration": call_session.duration,
                "call_status": call_session.status,
                "analysis": cached_block.get("analysis"),
                "model_used": cached_block.get("model_used"),
                "timestamp": cached_block.get("timestamp"),
                "is_cached": True
            }

        preferred_model: str | None = None
        agent = None
        agent_prompt: str | None = None

        if call_session.agent_id:
            try:
                agent = agent_service.get_agent_by_id(
                    db, call_session.agent_id, call_session.tenant_id
                )
                if agent:
                    # Get agent's system prompt (priority: agent.system_prompt > model.system_prompt)
                    if agent.system_prompt:
                        agent_prompt = agent.system_prompt
                        logger.debug(
                            "📝 Using agent's custom system prompt (%d chars)",
                            len(agent_prompt),
                        )
                    elif agent.model and agent.model.system_prompt:
                        agent_prompt = agent.model.system_prompt
                        logger.debug(
                            "📝 Using model's system prompt (%d chars)",
                            len(agent_prompt),
                        )

                    if agent and agent.model:
                        preferred_model = agent.model.model_name
                        logger.debug("🔍 Found call's model: %s", preferred_model)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "⚠️ Could not get call's model or agent prompt: %s", e
                )

        # Fallback models in priority order
        fallback_models: List[str] = [
            m
            for m in [
                preferred_model,
                "gemini-2.0-flash",
                "llama-3.3-70b-versatile",
                "gpt-4o-mini",
            ]
            if m
        ]

        model = None
        last_error: Exception | None = None

        # Try each model until one works (for presence)
        for model_name in fallback_models:
            try:
                logger.debug("🔄 Trying model: %s", model_name)
                model = self.model_service.get_model_by_name(db, model_name)
                if model:
                    logger.debug(
                        "✅ Model found: %s, Provider: %s",
                        model.model_name,
                        model.provider.name,
                    )
                    break
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("⚠️ Model %s not available: %s", model_name, e)
                last_error = e
                continue

        if not model:
            detail = f"No available model found. Tried: {', '.join(fallback_models)}"
            if last_error is not None:
                detail += f". Last error: {redact_pii(str(last_error))}"
            raise HTTPException(status_code=404, detail=detail)

        # Get transcript messages
        transcript_messages = transcript_service.get_messages_by_session(
            db, call_session.id
        )
        logger.debug(
            "🔍 Found %d transcript messages for session %s",
            len(transcript_messages),
            call_session.id,
        )

        if not transcript_messages:
            if raise_on_no_transcript:
                raise HTTPException(
                    status_code=404,
                    detail="No transcript messages found for this call session",
                )
            logger.info(
                "Skipping transcript analysis — no messages (session %s)",
                call_session.id,
            )
            return None

        # Format transcript for analysis
        transcript_text = ""
        for msg in transcript_messages:
            role_label = "Agent" if msg.role == "agent" else "Customer"
            transcript_text += f"{role_label}: {msg.message}\n"

        # Create analysis prompts
        summary_prompt = f"""
        Analyze this call transcript and provide a brief summary in 2-3 sentences.

        Call Transcript:
        {transcript_text}

        Provide only:
        - Brief call overview
        - Main topic/issue
        - Outcome/resolution

        Keep it concise and to the point.

        On the very last line only, add exactly one line in this format (for CRM display):
        CALLER_NAME: <name as stated by the customer> or CALLER_NAME: Unknown
        """

        sentiment_prompt = f"""
        Analyze the sentiment of this call transcript and provide a brief assessment.

        Call Transcript:
        {transcript_text}

        Provide only:
        - Overall sentiment (positive/negative/neutral)
        - Sentiment score (0-100)
        - Customer satisfaction level (high/medium/low)

        Keep it brief and concise.
        """

        outcome_prompt = f"""
Based only on this call transcript, judge whether the interaction achieved a successful outcome
for the customer and the agent's apparent goal (e.g. issue resolved, appointment booked, or not).

Call Transcript:
{transcript_text}

Reply in exactly two lines, no other text:
OUTCOME: success OR fail OR unclear
REASON: one short sentence (max 25 words)
"""

        # Create recommendations prompt based on agent's instructions
        recommendations_prompt = f"""
Analyze this call transcript and provide 2-3 brief, actionable recommendations for the agent.

Call Transcript:
{transcript_text}

Agent's Instructions/Purpose:
{agent_prompt if agent_prompt else "No specific instructions provided. Use general best practices for customer service calls."}

IMPORTANT - Keep recommendations BRIEF and CONCISE:
- Provide only 2-3 recommendations maximum
- Each recommendation should be 1 sentence only (brief and to the point)
- Be specific and actionable
- Use friendly, conversational tone

Format your response as:
1. [Brief recommendation in 1 sentence]
2. [Next brief recommendation in 1 sentence]
3. [Optional third recommendation in 1 sentence]

Keep it concise - similar to summary format. Maximum 1 sentence per recommendation.
"""

        # Perform analysis with automatic fallback on quota errors
        summary_result = None
        sentiment_result = None
        outcome_result = None
        recommendations_result = None
        used_model = None
        last_error_local: Exception | None = None

        for model_name in fallback_models:
            try:
                outcome_result = None
                current_model = self.model_service.get_model_by_name(db, model_name)
                if not current_model:
                    continue

                current_api_key = None
                if current_model.api_key:
                    from app.core.security import decrypt_api_key

                    current_api_key = decrypt_api_key(current_model.api_key)

                logger.debug(
                    "🔄 Attempting analysis with %s...", current_model.model_name
                )

                summary_result = generate_analysis_text(
                    current_model, current_api_key, summary_prompt, max_tokens=400
                )
                sentiment_result = generate_analysis_text(
                    current_model, current_api_key, sentiment_prompt, max_tokens=100
                )

                try:
                    outcome_result = generate_analysis_text(
                        current_model,
                        current_api_key,
                        outcome_prompt,
                        max_tokens=150,
                    )
                except Exception as oe:  # pragma: no cover - defensive
                    logger.warning(
                        "⚠️ Outcome classification failed on %s: %s",
                        current_model.model_name,
                        oe,
                    )
                    outcome_result = None

                if agent_prompt:
                    try:
                        recommendations_result = generate_analysis_text(
                            current_model,
                            current_api_key,
                            recommendations_prompt,
                            max_tokens=500,
                        )
                        logger.debug("✅ Recommendations generated")
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning(
                            "⚠️ Failed to generate recommendations: %s", e
                        )

                used_model = current_model.model_name
                logger.info("✅ Analysis successful with %s", used_model)
                break

            except Exception as e:  # pragma: no cover - defensive
                error_str = str(e)
                logger.warning("⚠️ Error with %s: %s", model_name, e)

                if (
                    "429" in error_str
                    or "quota" in error_str.lower()
                    or "exceeded" in error_str.lower()
                ):
                    logger.warning(
                        "⚠️ Quota exceeded for %s, trying next model...", model_name
                    )
                    last_error_local = e
                    continue
                else:
                    last_error_local = e
                    continue

        if not summary_result or not sentiment_result:
            error_msg = (
                f"Analysis failed with all models. Last error: {str(last_error_local)}"
            )
            logger.error("❌ %s", error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        raw_summary = summary_result["content"].strip()
        caller_name = "Unknown"
        summary_lines: List[str] = []
        for line in raw_summary.split("\n"):
            cm = re.match(r"^\s*CALLER_NAME:\s*(.+)\s*$", line, re.IGNORECASE)
            if cm:
                caller_name = (cm.group(1) or "").strip() or "Unknown"
                continue
            summary_lines.append(line)
        display_summary = "\n".join(summary_lines).strip()

        success_eval_llm: str | None = None
        if outcome_result:
            raw_o = (outcome_result.get("content") or "").strip()
            om = re.search(r"^\s*OUTCOME:\s*(\S+)", raw_o, re.MULTILINE | re.IGNORECASE)
            rm = re.search(r"^\s*REASON:\s*(.+)$", raw_o, re.MULTILINE | re.IGNORECASE)
            tag = om.group(1).strip().lower() if om else None
            reason = rm.group(1).strip() if rm else None
            if tag and reason:
                success_eval_llm = f"{tag} — {reason}"
            elif tag:
                success_eval_llm = tag
            elif raw_o:
                success_eval_llm = raw_o[:400]

        # Determine whether this call flow is HIPAA-enabled so analysis text
        # can be redacted before persistence.
        hipaa_enabled = False
        if call_session.call_flow_id:
            flow = db.query(CallFlow).filter(
                CallFlow.id == call_session.call_flow_id,
                CallFlow.is_deleted.is_(False),
            ).first()
            if flow:
                hipaa_enabled = bool(flow.hipaa_compliance)

        def _redact(text: str) -> str:
            return redact_phi_if_hipaa(text, hipaa_enabled=hipaa_enabled)

        analysis_data: Dict[str, Any] = {
            "summary": _redact(display_summary),
            "sentiment": _redact(sentiment_result["content"].strip()),
            "caller_name": _redact(caller_name),
            "success_evaluation": _redact(
                success_eval_llm or "unclear — could not classify outcome from transcript"
            ),
        }

        if recommendations_result:
            recommendations_text = recommendations_result["content"].strip()

            recommendations_list: List[str] = []
            lines = recommendations_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                match = re.match(r"^\d+\.\s*(.+)$", line)
                if match:
                    recommendations_list.append(_redact(match.group(1).strip()))
                elif line.startswith("- ") or line.startswith("* "):
                    recommendations_list.append(_redact(line[2:].strip()))
                elif len(line) > 20 and not recommendations_list:
                    recommendations_list.append(_redact(line))

            if not recommendations_list:
                recommendations_list = [_redact(recommendations_text)]

            analysis_data["recommendations"] = recommendations_list
            analysis_data["recommendations_text"] = _redact(recommendations_text)
        elif agent_prompt:
            analysis_data["recommendations"] = [
                "Unable to generate recommendations at this time."
            ]
            analysis_data["recommendations_text"] = (
                "Unable to generate recommendations at this time."
            )

        ts = datetime.now(timezone.utc).isoformat()
        result = {
            "call_session_id": str(call_session.id),
            "transcript_message_count": len(transcript_messages),
            "call_duration": call_session.duration,
            "call_status": call_session.status,
            "analysis": analysis_data,
            "model_used": used_model,
            "timestamp": ts,
        }

        analysis_block = {
            "analysis": analysis_data,
            "model_used": used_model,
            "timestamp": ts,
        }
        if call_session.call_metadata is None:
            call_session.call_metadata = {}
        call_session.call_metadata["llm_call_analysis"] = analysis_block
        flag_modified(call_session, "call_metadata")
        # Finalized, HIPAA-redacted summary — powers cross-session caller memory lookups.
        call_session.transcript_summary = analysis_data["summary"]
        db.add(call_session)

        log_row = db.query(CallLog).filter(CallLog.call_session_id == call_session.id).first()
        if log_row:
            if log_row.call_metadata is None:
                log_row.call_metadata = {}
            log_row.call_metadata["llm_call_analysis"] = analysis_block
            flag_modified(log_row, "call_metadata")
            db.add(log_row)

        db.commit()
        db.refresh(call_session)

        logger.info(
            "✅ Transcript analysis completed for session %s using %s",
            call_session.id,
            used_model,
        )

        return result


voice_analysis_service = VoiceAnalysisService()


def _pg_advisory_key_pair(key_uuid: uuid.UUID) -> Tuple[int, int]:
    """Two int32 keys derived from a UUID (stable per call session) for pg_advisory_lock.

    Uses the same derivation as inbound_call_crm_sync_service.py's call_log_id-keyed
    lock, but call_session_id and call_log_id are a different UUID column family, so
    a cross-collision is negligible at any realistic call volume. `pg_try_advisory_lock`
    is non-blocking, so a collision would never deadlock the two lock users against each
    other — worst case, one side sees "lock unavailable" and skips its one-shot work for
    that call (e.g. the summary job silently not firing) rather than racing.
    """
    return struct.unpack(">ii", key_uuid.bytes[:8])


def _try_acquire_call_summary_lock(db: Session, call_session_id: uuid.UUID) -> bool:
    """
    Serialize automatic call-summary generation for one call_session across workers
    (AI-side call-ending logic and Twilio's StatusCallback webhook can both schedule it).
    Returns False if lock unavailable (another worker is generating); caller should exit quietly.
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
        logger.warning("Call summary: advisory lock check failed (continuing without lock)", exc_info=True)
        return True


def _release_call_summary_lock(db: Session, call_session_id: uuid.UUID) -> None:
    try:
        if db.get_bind().dialect.name != "postgresql":
            return
        k1, k2 = _pg_advisory_key_pair(call_session_id)
        db.execute(text("SELECT pg_advisory_unlock(:k1, :k2)"), {"k1": k1, "k2": k2})
        db.commit()
    except Exception:
        logger.warning("Call summary: pg_advisory_unlock failed (non-critical)", exc_info=True)
    finally:
        try:
            db.rollback()
        except Exception:  # noqa: S110 - already logged the advisory-unlock failure above
            pass


def ensure_call_summary_locked(
    db: Session,
    call_session: CallSession,
    *,
    raise_on_no_transcript: bool = False,
) -> None:
    """
    Shared double-checked-locking helper for lazily computing call-summary
    analysis. Any caller that may run concurrently with the automatic
    `generate_call_summary` ARQ job (e.g. the post-call email summary job)
    should compute analysis through this helper instead of calling
    `analyze_call_transcript` directly, so both callers serialize on the same
    pg advisory lock keyed on call_session_id and never issue two paid LLM
    analysis calls for the same call.

    No-ops quietly if analysis is already cached, or if another worker
    currently holds the lock for this call_session_id (`pg_try_advisory_lock`
    is non-blocking).
    """
    if call_session.call_metadata and "llm_call_analysis" in call_session.call_metadata:
        return

    session_uuid = call_session.id
    if not _try_acquire_call_summary_lock(db, session_uuid):
        logger.debug(
            "Call summary generation: another worker is already generating the summary "
            "for this call, skipping session=%s",
            session_uuid,
        )
        return

    try:
        # Double-checked lock: force a fresh read past this worker's identity-map
        # cache, in case a concurrent winner committed while we waited for the
        # lock. Ordering invariant — call_session must not be locally mutated
        # between the caller's query and this refresh, or that mutation is lost.
        db.refresh(call_session)
        if call_session.call_metadata and "llm_call_analysis" in call_session.call_metadata:
            logger.debug(
                "Call summary generation skipped — cached by another worker while "
                "waiting for lock, session %s",
                session_uuid,
            )
            return

        voice_analysis_service.analyze_call_transcript(
            db,
            call_session,
            user_id=call_session.user_id,
            raise_on_no_transcript=raise_on_no_transcript,
        )
    finally:
        _release_call_summary_lock(db, session_uuid)


async def _generate_call_summary_arq_task(ctx: dict, call_session_id: str) -> None:
    """
    ARQ job entrypoint — registered as ``generate_call_summary`` in
    app/workers/batch_call_worker.py::WorkerSettings.functions.

    Runs the same LLM analysis as the dashboard's manual "analyze transcript"
    action, automatically once a call completes. `analyze_call_transcript`
    already no-ops cheaply if `call_metadata["llm_call_analysis"]` is cached,
    but two ARQ workers can race to pass that check before either commits —
    a pg advisory lock keyed on call_session_id ensures only one worker
    actually runs the (paid) LLM analysis.
    """
    db: Session = SessionLocal()
    try:
        session_uuid = uuid.UUID(call_session_id)
        call_session = (
            db.query(CallSession).filter(CallSession.id == session_uuid).first()
        )
        if not call_session:
            logger.warning(
                "Call summary generation skipped — session not found: %s",
                call_session_id,
            )
            return

        if call_session.call_metadata and "llm_call_analysis" in call_session.call_metadata:
            logger.debug(
                "Call summary generation skipped — already cached for session %s",
                call_session_id,
            )
            return

        ensure_call_summary_locked(db, call_session, raise_on_no_transcript=False)
    except Exception:
        logger.warning(
            "Automatic call summary generation failed (non-critical) session=%s",
            call_session_id,
            exc_info=True,
        )
    finally:
        db.close()


def schedule_call_summary_generation(call_session_id: uuid.UUID) -> None:
    """
    Enqueue automatic call summary/sentiment/recommendations generation as an
    ARQ background job. Fire-and-forget — never blocks the caller. Fails open
    if the ARQ pool isn't ready: summary generation is best-effort and must
    never affect call completion.
    """
    pool = get_arq_pool()
    if pool is None:
        logger.warning(
            "ARQ pool not ready; call summary generation skipped for session=%s",
            call_session_id,
        )
        return

    async def _enqueue() -> None:
        try:
            await pool.enqueue_job("generate_call_summary", str(call_session_id))
        except Exception as exc:
            logger.warning("Failed to enqueue call summary generation job: %s", exc)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_enqueue())
        return
    asyncio.create_task(_enqueue())

