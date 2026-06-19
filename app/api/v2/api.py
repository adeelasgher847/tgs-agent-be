from fastapi import APIRouter

from app.api.v2.routers.active_calls import router as active_calls_router
from app.api.v2.routers.audit_events import router as audit_events_router
from app.api.v2.routers.batch_calls import router as batch_calls_router
from app.api.v2.routers.health import router as health_router
from app.api.v2.routers.webhooks import router as webhooks_router
from app.api.v2.routers.callback_scheduler import agents_router as cb_agents_router
from app.api.v2.routers.callback_scheduler import calls_router as cb_calls_router
from app.api.v2.routers.workspace import v2_router as workspace_router
from app.api.v2.routers.hipaa import flows_router as hipaa_flows_router
from app.api.v2.routers.hipaa import workspace_router as hipaa_workspace_router
from app.api.v2.routers.workspace import router as workspace_gdpr_router

v2_router = APIRouter()
v2_router.include_router(health_router)
v2_router.include_router(active_calls_router)
v2_router.include_router(audit_events_router)
v2_router.include_router(batch_calls_router)
v2_router.include_router(webhooks_router)
v2_router.include_router(cb_agents_router)
v2_router.include_router(cb_calls_router)
v2_router.include_router(workspace_router, prefix="/workspace", tags=["Workspace Settings"])
v2_router.include_router(hipaa_flows_router)
v2_router.include_router(hipaa_workspace_router)
v2_router.include_router(workspace_gdpr_router)
