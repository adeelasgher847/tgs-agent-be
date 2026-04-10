"""
On-demand appointment intake briefing from call transcript (demo: not persisted).

Separate from VoiceAnalysisService / call log analysis: different prompt, no sentiment scores.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.appointment import Appointment
from app.services.agent_service import agent_service
from app.services.model_service import ModelService
from app.services.transcript_service import transcript_service


_JSON_SYSTEM = """You extract structured intake information for front-desk staff from a phone call transcript.
Output ONLY valid JSON (no markdown fences, no commentary). Use this exact schema:
{
  "reason_for_visit": string or null,
  "health_symptoms_or_conditions": string or null,
  "customer_details_mentioned": string or null,
  "staff_briefing": string or null,
  "key_points": string or null
}

Rules:
- Write clear, professional English (or match the transcript language if not English).
- reason_for_visit: why they booked / main purpose (1–3 sentences max).
- health_symptoms_or_conditions: only facts the customer stated about symptoms, conditions, pain, or health concerns; null if not discussed.
- customer_details_mentioned: other useful facts (availability, preferences, family member, insurance mentioned, etc.); null if none.
- staff_briefing: 2–5 sentences the team should read before the appointment; null if nothing beyond reason_for_visit.
- key_points: concise scan-ready summary points in one variable/string (not array). Use semicolons to separate points. Null if nothing useful.
- Do NOT invent medical facts; if unsure, say so briefly or use null.
- Do NOT include customer_name, customer_phone, customer_email, appointment_reason, duration_minutes, or lifecycle status in these five intake fields.
- Do NOT include: sentiment labels, sentiment scores, satisfaction scores, star ratings, NPS, or emotional analytics.
- Do NOT output any keys other than the five above."""


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty model response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()
    try:
        out = json.loads(text)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("Could not parse JSON from model response")


def _allowed_only(data: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "reason_for_visit",
        "health_symptoms_or_conditions",
        "customer_details_mentioned",
        "staff_briefing",
        "key_points",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if k not in data:
            continue
        v = data[k]
        if k == "key_points":
            if v is None:
                out[k] = None
            elif isinstance(v, list):
                points = [str(x).strip() for x in v if str(x).strip()][:20]
                out[k] = "; ".join(points) if points else None
            else:
                text_v = str(v).strip()
                out[k] = text_v if text_v else None
        else:
            if v is None or (isinstance(v, str) and not v.strip()):
                out[k] = None
            else:
                out[k] = str(v).strip()
    return out


def _generate_with_provider(
    *,
    provider_name: str,
    model_name: str,
    api_key: Optional[str],
    user_prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    pn = (provider_name or "").strip().lower()
    if pn in (
        "gemini",
        "google",
        "google-ai",
        "google ai",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
    ):
        from app.services.gemini_service import GeminiService

        return GeminiService().generate_text(
            prompt=user_prompt,
            system_prompt=_JSON_SYSTEM,
            model_name=model_name,
            temperature=0.2,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    if pn in ("openai", "gpt", "gpt-4o-mini", "gpt-4o", "gpt-4"):
        from app.services.openai_service import OpenAIService

        return OpenAIService().generate_text(
            prompt=user_prompt,
            system_prompt=_JSON_SYSTEM,
            model_name=model_name,
            temperature=0.2,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    if pn in ("groq", "llama", "llama-3.3-70b-versatile"):
        from app.services.groq_service import GroqService

        return GroqService().generate_text(
            prompt=user_prompt,
            system_prompt=_JSON_SYSTEM,
            model_name=model_name,
            temperature=0.2,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    raise ValueError(f"Unsupported provider: {provider_name}")


class AppointmentIntakeSummaryService:
    def __init__(self) -> None:
        self.model_service = ModelService()

    def build_user_prompt(
        self,
        *,
        transcript_text: str,
        appointment: Appointment,
    ) -> str:
        appt_bits = [
            f"Appointment reason (from booking record): {appointment.appointment_reason or '—'}",
            f"Customer name (record): {appointment.customer_name}",
            f"Customer phone (record): {appointment.customer_phone}",
        ]
        if appointment.customer_email:
            appt_bits.append(f"Customer email (record): {appointment.customer_email}")
        return (
            "Use the booking record for names/contact when the transcript is unclear.\n\n"
            + "\n".join(appt_bits)
            + "\n\nCall transcript (Customer / Agent):\n"
            + transcript_text
        )

    def generate_intake_summary(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        appointment: Appointment,
    ) -> Dict[str, Any]:
        if not appointment.call_session_id:
            raise HTTPException(
                status_code=422,
                detail="This appointment is not linked to a call; no transcript is available.",
            )

        messages = transcript_service.get_messages_by_session(
            db, appointment.call_session_id
        )
        if not messages:
            raise HTTPException(
                status_code=404,
                detail="No transcript messages found for this appointment's call session.",
            )

        transcript_lines: List[str] = []
        for msg in messages:
            role_label = "Agent" if msg.role == "agent" else "Customer"
            transcript_lines.append(f"{role_label}: {msg.message}")
        transcript_text = "\n".join(transcript_lines)
        if len(transcript_text) > 120_000:
            transcript_text = transcript_text[-120_000:]

        user_prompt = self.build_user_prompt(
            transcript_text=transcript_text, appointment=appointment
        )

        preferred_model: Optional[str] = None
        if appointment.agent_id:
            try:
                agent = agent_service.get_agent_by_id(
                    db, appointment.agent_id, tenant_id
                )
                if agent and agent.model:
                    preferred_model = agent.model.model_name
            except Exception as e:
                logger.warning("Intake summary: could not load agent model: %s", e)

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

        last_error: Optional[Exception] = None
        used_model: Optional[str] = None
        raw_content: Optional[str] = None

        for model_name in fallback_models:
            try:
                current = self.model_service.get_model_by_name(db, model_name)
                if not current:
                    continue
                api_key = None
                if current.api_key:
                    from app.core.security import decrypt_api_key

                    api_key = decrypt_api_key(current.api_key)

                prov = getattr(current, "provider", None)
                provider_name = (getattr(prov, "name", None) or "").strip()
                if not provider_name:
                    continue

                result = _generate_with_provider(
                    provider_name=provider_name,
                    model_name=current.model_name,
                    api_key=api_key,
                    user_prompt=user_prompt,
                    max_tokens=1800,
                )
                raw_content = (result.get("content") or "").strip()
                if not raw_content:
                    continue
                used_model = current.model_name
                break
            except Exception as e:
                err = str(e).lower()
                last_error = e
                if (
                    "429" in str(e)
                    or "quota" in err
                    or "exceeded" in err
                    or "rate" in err
                ):
                    logger.warning(
                        "Intake summary: model %s unavailable (%s), trying next",
                        model_name,
                        e,
                    )
                    continue
                logger.warning("Intake summary: model %s failed: %s", model_name, e)
                continue

        if not raw_content or not used_model:
            detail = (
                f"Could not generate intake summary. Last error: {last_error!s}"
                if last_error
                else "No LLM model available."
            )
            raise HTTPException(status_code=503, detail=detail)

        try:
            parsed = _extract_json_object(raw_content)
        except Exception as e:
            logger.error("Intake summary: JSON parse failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=502,
                detail="The model returned invalid JSON; try again.",
            ) from e

        intake = _allowed_only(parsed)

        return {
            "appointment_id": appointment.id,
            "call_session_id": appointment.call_session_id,
            "customer_name": appointment.customer_name,
            "customer_phone": appointment.customer_phone,
            "customer_email": appointment.customer_email,
            "appointment_reason": appointment.appointment_reason,
            "duration_minutes": appointment.duration_minutes,
            "status": appointment.status,
            "review_status": (appointment.review_status or "not_reviewed"),
            "reviewed_at": appointment.reviewed_at,
            "generated_at": datetime.now(timezone.utc),
            "model_used": used_model,
            "transcript_message_count": len(messages),
            "intake": intake,
        }


appointment_intake_summary_service = AppointmentIntakeSummaryService()
