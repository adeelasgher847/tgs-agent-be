from fastapi import APIRouter

from app.api.v2.routers.health import router as health_router

v2_router = APIRouter()
v2_router.include_router(health_router)
