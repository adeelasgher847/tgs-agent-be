from fastapi import APIRouter
from app.api.api_v1.endpoints import user, tenant, role
from app.routers.agent import router as agent_router

api_router = APIRouter()
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(tenant.router, prefix="/tenants", tags=["tenants"])
api_router.include_router(role.router, prefix="/roles", tags=["roles"]) 
api_router.include_router(agent_router, prefix="/agent", tags=["Voice Agent"]) 