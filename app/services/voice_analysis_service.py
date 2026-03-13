from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session

from fastapi import HTTPException

from app.core.logger import logger
from app.models.call_session import CallSession
from app.services.agent_service import agent_service
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
    ) -> Dict[str, Any]:
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
            raise HTTPException(
                status_code=404,
                detail="No transcript messages found for this call session",
            )

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
        recommendations_result = None
        used_model = None
        last_error_local: Optional[Exception] = None

        for model_name in fallback_models:
            try:
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

        analysis_data: Dict[str, Any] = {
            "summary": summary_result["content"].strip(),
            "sentiment": sentiment_result["content"].strip(),
        }

        if recommendations_result:
            recommendations_text = recommendations_result["content"].strip()
            import re

            recommendations_list: List[str] = []
            lines = recommendations_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                match = re.match(r"^\d+\.\s*(.+)$", line)
                if match:
                    recommendations_list.append(match.group(1).strip())
                elif line.startswith("- ") or line.startswith("* "):
                    recommendations_list.append(line[2:].strip())
                elif len(line) > 20 and not recommendations_list:
                    recommendations_list.append(line)

            if not recommendations_list:
                recommendations_list = [recommendations_text]

            analysis_data["recommendations"] = recommendations_list
            analysis_data["recommendations_text"] = recommendations_text
        elif agent_prompt:
            analysis_data["recommendations"] = [
                "Unable to generate recommendations at this time."
            ]
            analysis_data["recommendations_text"] = (
                "Unable to generate recommendations at this time."
            )

        result = {
            "call_session_id": str(call_session.id),
            "transcript_message_count": len(transcript_messages),
            "call_duration": call_session.duration,
            "call_status": call_session.status,
            "analysis": analysis_data,
            "model_used": used_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "✅ Transcript analysis completed for session %s using %s",
            call_session.id,
            used_model,
        )

        return result


voice_analysis_service = VoiceAnalysisService()

