from fastapi import APIRouter

from app.api.api_v1.endpoints import (
    accept_invite,
    allowed_domains,
    api_keys,
    billing,
    gemini,
    model,
    openai,
    plan,
    provider,
    role,
    tenant,
    user,
    workspace,
    workspace_invites,
)
from app.routers.agents import router as agent_router
from app.routers.call_flows import router as call_flows_router
from app.routers.folders import router as folders_router
from app.routers.bidirectional_stream import router as bidirectional_stream_router
from app.routers.livekit_bridge import router as livekit_bridge_router
from app.routers.call_logs import router as call_logs_router
from app.routers.call_sessions import router as call_sessions_router
from app.routers.clickup_oauth import router as clickup_oauth_router
from app.routers.crm_config import router as crm_config_router
from app.routers.job_description import router as job_description_router
from app.routers.knowledge_base import router as knowledge_base_router
from app.routers.phone_numbers import router as phone_numbers_router
from app.routers.telephony import router as telephony_router
from app.routers.transfer_routes import router as transfer_routes_router
from app.routers.recruitment_dashboard import router as recruitment_dashboard_router
from app.routers.resumes import router as resume_router
from app.routers.resume_interviews import router as resume_interviews_router
from app.routers.scheduled_calls import router as scheduled_calls_router
from app.routers.sdk import router as sdk_router
from app.routers.tts_audio import router as tts_audio_router
from app.routers.tts import router as tts_router
from app.routers.voice import router as voice_router
from app.routers.voice_gather import router as voice_gather_router
from app.routers.live_voice import router as live_voice_router
from app.routers.general_websocket import router as general_websocket_router
from app.routers.calendar import router as calendar_router
from app.routers.inbound_crm import router as inbound_crm_router
from app.routers.internal_tts import router as internal_tts_router
from app.routers.internal_stt import router as internal_stt_router
from app.routers.business_knowledge import router as business_knowledge_router
from app.routers.recordings import router as recordings_router
from app.routers.integrations import router as integrations_router
from app.routers.hubspot_integration import router as hubspot_integration_router
from app.routers.call_history import router as call_history_router
from app.routers.call_history import batch_router as batch_call_metrics_router

api_router = APIRouter()
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(api_keys.router, prefix="/api-keys", tags=["API Keys"])
api_router.include_router(tenant.router, prefix="/tenants", tags=["tenants"])
# Invite/allowed-domains sub-routes must be registered BEFORE workspace.router so
# that their literal paths take priority over /workspace/{workspace_id}.
api_router.include_router(workspace_invites.router, prefix="/workspace", tags=["Workspace Invitations"])
api_router.include_router(allowed_domains.router, prefix="/workspace", tags=["Workspace — Allowed Domains"])
api_router.include_router(workspace.router, prefix="/workspace", tags=["Workspace"])
api_router.include_router(role.router, prefix="/roles", tags=["roles"])
api_router.include_router(agent_router, prefix="/agent", tags=["Voice Agent"])
api_router.include_router(call_flows_router, prefix="/call-flows", tags=["Call Flows"])
api_router.include_router(folders_router, prefix="/folders", tags=["Folders"])
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
api_router.include_router(telephony_router, prefix="/telephony", tags=["Telephony"])
api_router.include_router(
    transfer_routes_router,
    prefix="/transfer-routes",
    tags=["Transfer routes"],
)
api_router.include_router(call_sessions_router, prefix="/call-sessions", tags=["Call Sessions"])
api_router.include_router(call_logs_router, prefix="/call-logs", tags=["Call Logs"])
api_router.include_router(general_websocket_router, prefix="/general", tags=["General WebSocket"])
api_router.include_router(accept_invite.router, prefix="/accept-invite", tags=["accept-invite"])
api_router.include_router(billing.router, prefix="/billing", tags=["billing"])
api_router.include_router(plan.router, prefix="/plans", tags=["plans"])
api_router.include_router(provider.router, prefix="/providers", tags=["providers"], include_in_schema=True)
api_router.include_router(model.router, prefix="/models", tags=["models"])
api_router.include_router(gemini.router, prefix="/gemini", tags=["gemini"], include_in_schema=False)
api_router.include_router(openai.router, prefix="/openai", tags=["openai"], include_in_schema=False)
api_router.include_router(tts_audio_router, prefix="/tts", tags=["Google TTS"], include_in_schema=False)
api_router.include_router(tts_router, prefix="/tts", tags=["TTS"])
api_router.include_router(internal_tts_router, prefix="/internal/tts", tags=["Internal TTS"])
api_router.include_router(internal_stt_router, prefix="/internal/stt", tags=["Internal STT"])
api_router.include_router(
    bidirectional_stream_router,
    prefix="/stream",
    tags=["Bidirectional Streaming"],
    include_in_schema=False,
)
api_router.include_router(
    livekit_bridge_router,
    prefix="/livekit",
    tags=["LiveKit Bridge"],
    include_in_schema=False,
)
api_router.include_router(scheduled_calls_router, prefix="/schedule", tags=["Scheduled Calls"])
api_router.include_router(sdk_router, prefix="/sdk", tags=["Web SDK — Public"])
api_router.include_router(crm_config_router, prefix="/crm-config", tags=["CRM Configuration"])
api_router.include_router(
    clickup_oauth_router,
    prefix="/auth/clickup",
    tags=["ClickUp OAuth"],
    include_in_schema=False,
)
api_router.include_router(knowledge_base_router, prefix="/kb", tags=["Knowledge Base"])
api_router.include_router(
    business_knowledge_router,
    prefix="/business-knowledge",
    tags=["Business Knowledge"],
)
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
api_router.include_router(
    recruitment_dashboard_router,
    prefix="/recruiting/dashboard",
    tags=["Recruiting Dashboard"],
)
api_router.include_router(recordings_router, prefix="/recordings", tags=["Call Recordings"])
api_router.include_router(integrations_router, prefix="/integrations", tags=["Integrations"])
api_router.include_router(
    hubspot_integration_router,
    prefix="/integrations/hubspot",
    tags=["HubSpot Integration"],
)
api_router.include_router(call_history_router, prefix="/calls", tags=["Call History Analytics"])
api_router.include_router(batch_call_metrics_router, prefix="/batch-calls", tags=["Batch Call Analytics"])
