from fastapi import APIRouter
from app.api.api_v1.endpoints import user, tenant, role, invite, accept_invite, billing, plan
from app.routers.agent import router as agent_router
from app.routers.voice import router as voice_router
from app.routers.live_voice import router as live_voice_router
from app.routers.phone_numbers import router as phone_numbers_router
from app.routers.call_sessions import router as call_sessions_router
from app.routers.call_logs import router as call_logs_router

api_router = APIRouter()
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(tenant.router, prefix="/tenants", tags=["tenants"])
api_router.include_router(role.router, prefix="/roles", tags=["roles"]) 
api_router.include_router(agent_router, prefix="/agent", tags=["Voice Agent"])
api_router.include_router(voice_router, prefix="/voice", tags=["Voice Calls"])
api_router.include_router(live_voice_router, prefix="/live-voice", tags=["Live Voice - Talk to Assistant"])
api_router.include_router(phone_numbers_router, prefix="/phone-numbers", tags=["Phone Numbers"])
api_router.include_router(call_sessions_router, prefix="/call-sessions", tags=["Call Sessions"])
api_router.include_router(call_logs_router, prefix="/call-logs", tags=["Call Logs"])
api_router.include_router(invite.router, prefix="/invites", tags=["invites"])
api_router.include_router(accept_invite.router, prefix="/accept-invite", tags=["accept-invite"])
api_router.include_router(billing.router, prefix="/billing", tags=["billing"])
api_router.include_router(plan.router, prefix="/plans", tags=["plans"])
