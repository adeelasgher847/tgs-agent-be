from fastapi import APIRouter

from app.api.v2.routers.active_calls import router as active_calls_router
from app.api.v2.routers.batch_calls import router as batch_calls_router
from app.api.v2.routers.health import router as health_router
from app.api.v2.routers.webhooks import router as webhooks_router
from app.api.v2.routers.callback_scheduler import agents_router as cb_agents_router
from app.api.v2.routers.callback_scheduler import calls_router as cb_calls_router

v2_router = APIRouter()
v2_router.include_router(health_router)
v2_router.include_router(active_calls_router)
v2_router.include_router(batch_calls_router)
v2_router.include_router(webhooks_router)
v2_router.include_router(cb_agents_router)
v2_router.include_router(cb_calls_router)
