from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.agent import Agent
from app.models.call_session import CallSession


class VoiceAnalyticsService:
    """Service for computing voice/call analytics for the dashboard."""

    def get_dashboard_analytics(
        self,
        db: Session,
        tenant_id,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.debug(
            "Computing dashboard analytics for tenant %s (agent_id=%s)",
            str(tenant_id),
            agent_id,
        )

        base_query = db.query(CallSession).filter(CallSession.tenant_id == tenant_id)

        if agent_id:
            from uuid import UUID

            agent_uuid = UUID(agent_id)
            base_query = base_query.filter(CallSession.agent_id == agent_uuid)

        call_sessions: List[CallSession] = base_query.all()

        total_calls = len(call_sessions)

        completed_calls = [
            call
            for call in call_sessions
            if call.status == "completed" and call.duration is not None
        ]

        if completed_calls:
            total_duration = sum(call.duration for call in completed_calls)
            average_duration = total_duration / len(completed_calls)
        else:
            average_duration = 0

        status_counts: Dict[str, int] = {}
        for call in call_sessions:
            status = call.status or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1

        type_counts: Dict[str, int] = {}
        for call in call_sessions:
            call_type = call.call_type or "unknown"
            type_counts[call_type] = type_counts.get(call_type, 0) + 1

        agent_stats: Dict[str, Any] = {}
        if not agent_id:
            agents: List[Agent] = db.query(Agent).filter(Agent.tenant_id == tenant_id).all()

            for agent in agents:
                agent_calls = [call for call in call_sessions if call.agent_id == agent.id]
                agent_completed = [
                    call
                    for call in agent_calls
                    if call.status == "completed" and call.duration is not None
                ]

                agent_avg_duration = 0
                if agent_completed:
                    agent_total_duration = sum(call.duration for call in agent_completed)
                    agent_avg_duration = agent_total_duration / len(agent_completed)

                agent_stats[str(agent.id)] = {
                    "agent_name": agent.name,
                    "total_calls": len(agent_calls),
                    "completed_calls": len(agent_completed),
                    "average_duration_seconds": round(agent_avg_duration, 2),
                    "average_duration_minutes": round(agent_avg_duration / 60, 2),
                }

        recent_calls = (
            base_query.order_by(CallSession.created_at.desc()).limit(10).all()
        )

        recent_calls_data = []
        for call in recent_calls:
            recent_calls_data.append(
                {
                    "id": str(call.id),
                    "call_sid": call.twilio_call_sid,
                    "agent_name": call.agent.name if call.agent else "Unknown",
                    "status": call.status,
                    "call_type": call.call_type,
                    "duration": call.duration,
                    "start_time": call.start_time.isoformat()
                    if call.start_time
                    else None,
                    "end_time": call.end_time.isoformat() if call.end_time else None,
                    "from_number": call.from_number,
                    "to_number": call.to_number,
                    "cost": call.cost,
                    "recording_url": call.recording_url,
                    "has_recording": call.recording_url is not None,
                }
            )

        analytics_data: Dict[str, Any] = {
            "tenant_id": str(tenant_id),
            "filtered_by_agent": agent_id is not None,
            "agent_id": agent_id,
            "total_calls": total_calls,
            "completed_calls": len(completed_calls),
            "average_duration_seconds": round(average_duration, 2),
            "average_duration_minutes": round(average_duration / 60, 2),
            "status_breakdown": status_counts,
            "call_type_breakdown": type_counts,
            "agent_statistics": agent_stats,
            "recent_calls": recent_calls_data,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return analytics_data


voice_analytics_service = VoiceAnalyticsService()

