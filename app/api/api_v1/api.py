from fastapi import APIRouter
from app.api.api_v1.endpoints import user, tenant, role, invite, accept_invite, billing, plan, provider, model, gemini, openai
from app.routers.agent import router as agent_router
from app.routers.voice import router as voice_router
from app.routers.voice_gather import router as voice_gather_router
from app.routers.live_voice import router as live_voice_router
from app.routers.phone_numbers import router as phone_numbers_router
from app.routers.call_sessions import router as call_sessions_router
from app.routers.call_logs import router as call_logs_router
from app.routers.general_websocket import router as general_websocket_router
from app.routers.tts_audio import router as tts_audio_router
from app.routers.bidirectional_stream import router as bidirectional_stream_router
from app.routers.scheduled_calls import router as scheduled_calls_router
from app.routers.crm_config import router as crm_config_router
from app.routers.clickup_oauth import router as clickup_oauth_router
from app.routers.knowledge_base import router as knowledge_base_router
from app.routers.calendar import router as calendar_router

api_router = APIRouter()
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(tenant.router, prefix="/tenants", tags=["tenants"])
api_router.include_router(role.router, prefix="/roles", tags=["roles"]) 
api_router.include_router(agent_router, prefix="/agent", tags=["Voice Agent"])
api_router.include_router(voice_router, prefix="/voice", tags=["Voice Calls"])
api_router.include_router(voice_gather_router, prefix="/voice", tags=["Voice Calls - Gather"], include_in_schema=False)
api_router.include_router(live_voice_router, prefix="/live-voice", tags=["Live Voice - Talk to Assistant"],include_in_schema=False)
api_router.include_router(phone_numbers_router, prefix="/phone-numbers", tags=["Phone Numbers"])
api_router.include_router(call_sessions_router, prefix="/call-sessions", tags=["Call Sessions"])
api_router.include_router(call_logs_router, prefix="/call-logs", tags=["Call Logs"])
api_router.include_router(general_websocket_router, prefix="/general", tags=["General WebSocket"])
api_router.include_router(invite.router, prefix="/invites", tags=["invites"])
api_router.include_router(accept_invite.router, prefix="/accept-invite", tags=["accept-invite"])
api_router.include_router(billing.router, prefix="/billing", tags=["billing"])
api_router.include_router(plan.router, prefix="/plans", tags=["plans"])
api_router.include_router(provider.router, prefix="/providers", tags=["providers"],include_in_schema=True)
api_router.include_router(model.router, prefix="/models", tags=["models"])
api_router.include_router(gemini.router, prefix="/gemini", tags=["gemini"],include_in_schema=False)
api_router.include_router(openai.router, prefix="/openai", tags=["openai"],include_in_schema=False)
api_router.include_router(tts_audio_router, prefix="/tts", tags=["Google TTS"], include_in_schema=False)
api_router.include_router(bidirectional_stream_router, prefix="/stream", tags=["Bidirectional Streaming"], include_in_schema=False)
api_router.include_router(scheduled_calls_router, prefix="/schedule", tags=["Scheduled Calls"])
api_router.include_router(crm_config_router, prefix="/crm-config", tags=["CRM Configuration"])
api_router.include_router(clickup_oauth_router, prefix="/auth/clickup", tags=["ClickUp OAuth"],include_in_schema=False)
api_router.include_router(knowledge_base_router, prefix="/kb", tags=["Knowledge Base"])
api_router.include_router(calendar_router, prefix="/calendar", tags=["Calendar"])
