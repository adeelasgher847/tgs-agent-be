from fastapi import APIRouter

from app.api.api_v1.endpoints import (
    accept_invite,
    billing,
    gemini,
    invite,
    model,
    openai,
    plan,
    provider,
    role,
    tenant,
    user,
)
from app.routers.agent import router as agent_router
from app.routers.bidirectional_stream import router as bidirectional_stream_router
from app.routers.call_logs import router as call_logs_router
from app.routers.call_sessions import router as call_sessions_router
from app.routers.clickup_oauth import router as clickup_oauth_router
from app.routers.crm_config import router as crm_config_router
from app.routers.job_description import router as job_description_router
from app.routers.knowledge_base import router as knowledge_base_router
from app.routers.phone_numbers import router as phone_numbers_router
from app.routers.resume import router as resume_router
from app.routers.resume_interviews import router as resume_interviews_router
from app.routers.scheduled_calls import router as scheduled_calls_router
from app.routers.tts_audio import router as tts_audio_router
from app.routers.tts import router as tts_router
from app.routers.voice import router as voice_router
from app.routers.voice_gather import router as voice_gather_router
from app.routers.live_voice import router as live_voice_router
from app.routers.general_websocket import router as general_websocket_router
from app.routers.calendar import router as calendar_router
from app.routers.inbound_crm import router as inbound_crm_router
from app.routers.internal_tts import router as internal_tts_router

api_router = APIRouter()
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(tenant.router, prefix="/tenants", tags=["tenants"])
api_router.include_router(role.router, prefix="/roles", tags=["roles"])
api_router.include_router(agent_router, prefix="/agent", tags=["Voice Agent"])
api_router.include_router(voice_router, prefix="/voice", tags=["Voice Calls"])
api_router.include_router(
    voice_gather_router,
    prefix="/voice",
    tags=["Voice Calls - Gather"],
    include_in_schema=False,
)
api_router.include_router(
    live_voice_router,
    prefix="/live-voice",
    tags=["Live Voice - Talk to Assistant"],
    include_in_schema=False,
)
api_router.include_router(phone_numbers_router, prefix="/phone-numbers", tags=["Phone Numbers"])
api_router.include_router(call_sessions_router, prefix="/call-sessions", tags=["Call Sessions"])
api_router.include_router(call_logs_router, prefix="/call-logs", tags=["Call Logs"])
api_router.include_router(general_websocket_router, prefix="/general", tags=["General WebSocket"])
api_router.include_router(invite.router, prefix="/invites", tags=["invites"])
api_router.include_router(accept_invite.router, prefix="/accept-invite", tags=["accept-invite"])
api_router.include_router(billing.router, prefix="/billing", tags=["billing"])
api_router.include_router(plan.router, prefix="/plans", tags=["plans"])
api_router.include_router(provider.router, prefix="/providers", tags=["providers"], include_in_schema=True)
api_router.include_router(model.router, prefix="/models", tags=["models"])
api_router.include_router(gemini.router, prefix="/gemini", tags=["gemini"], include_in_schema=False)
api_router.include_router(openai.router, prefix="/openai", tags=["openai"], include_in_schema=False)
api_router.include_router(tts_audio_router, prefix="/tts", tags=["Google TTS"], include_in_schema=True)
api_router.include_router(tts_router, prefix="/tts", tags=["TTS"])
api_router.include_router(internal_tts_router, prefix="/internal/tts", tags=["Internal TTS"])
api_router.include_router(
    bidirectional_stream_router,
    prefix="/stream",
    tags=["Bidirectional Streaming"],
    include_in_schema=False,
)
api_router.include_router(scheduled_calls_router, prefix="/schedule", tags=["Scheduled Calls"])
api_router.include_router(crm_config_router, prefix="/crm-config", tags=["CRM Configuration"])
api_router.include_router(
    clickup_oauth_router,
    prefix="/auth/clickup",
    tags=["ClickUp OAuth"],
    include_in_schema=False,
)
api_router.include_router(knowledge_base_router, prefix="/kb", tags=["Knowledge Base"])
api_router.include_router(calendar_router, prefix="/calendar", tags=["Calendar"])
api_router.include_router(inbound_crm_router, prefix="/inbound-crm", tags=["Inbound CRM — Call logs"])
api_router.include_router(
    job_description_router,
    prefix="/recruiting/job-descriptions",
    tags=["Recruiting Job Descriptions"],
)
api_router.include_router(
    resume_router,
    prefix="/recruiting/resumes",
    tags=["Recruiting Resumes"],
)
# Alias: singular /resume (some clients used this path; same routes as above)
api_router.include_router(
    resume_router,
    prefix="/recruiting/resume",
    tags=["Recruiting Resumes (alias)"],
    include_in_schema=False,
)
api_router.include_router(
    resume_interviews_router,
    prefix="/recruiting/resume-interviews",
    tags=["Recruiting Resume Interviews"],
)
