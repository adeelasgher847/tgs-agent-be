import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from fastapi import HTTPException

from app.core.logger import logger
from app.models.call_flow import CallFlow
from app.models.call_log import CallLog
from app.models.call_session import CallSession
from app.services.agent_service import agent_service
from app.services.dlp_service import redact_phi_if_hipaa
from app.services.model_service import ModelService
from app.services.transcript_service import transcript_service
from app.utils.response import create_success_response


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
    ) -> Optional[Dict[str, Any]]:
        """Behavior-preserving refactor of `analyze_call_transcript` logic from `voice.py`."""
        from uuid import UUID

        # Check if user has access to this call session
        # (same logic as original route)
        # NOTE: caller is responsible for validating call_session_id format and existence.
        from app.models.user import User  # type: ignore

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

        preferred_model: Optional[str] = None
        agent = None
        agent_prompt: Optional[str] = None

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
        last_error: Optional[Exception] = None

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
            raise HTTPException(
                status_code=404,
                detail=f"No available model found. Tried: {', '.join(fallback_models)}",
            )

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

        # Helper function to call appropriate service based on provider
        def generate_analysis_text(current_model, current_api_key, prompt: str, max_tokens: int = 200):
            """Generate text using the appropriate service based on provider."""
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

        # Perform analysis with automatic fallback on quota errors
        summary_result = None
        sentiment_result = None
        outcome_result = None
        recommendations_result = None
        used_model = None
        last_error_local: Optional[Exception] = None

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
                    current_model, current_api_key, summary_prompt, max_tokens=200
                )
                sentiment_result = generate_analysis_text(
                    current_model, current_api_key, sentiment_prompt, max_tokens=150
                )

                try:
                    outcome_result = generate_analysis_text(
                        current_model,
                        current_api_key,
                        outcome_prompt,
                        max_tokens=120,
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
                            max_tokens=300,
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

        success_eval_llm: Optional[str] = None
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

