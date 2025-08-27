from fastapi import APIRouter
from app.api.api_v1.endpoints import user, tenant, role
from app.routers.agent import router as agent_router
from app.routers.voice import router as voice_router
from app.routers.phone_numbers import router as phone_numbers_router
from app.routers.call_sessions import router as call_sessions_router

api_router = APIRouter()
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(tenant.router, prefix="/tenants", tags=["tenants"])
api_router.include_router(role.router, prefix="/roles", tags=["roles"]) 
api_router.include_router(agent_router, prefix="/agent", tags=["Voice Agent"])
api_router.include_router(voice_router, prefix="/voice", tags=["Voice Calls"])
api_router.include_router(phone_numbers_router, prefix="/phone-numbers", tags=["Phone Numbers"])
api_router.include_router(call_sessions_router, prefix="/call-sessions", tags=["Call Sessions"]) 