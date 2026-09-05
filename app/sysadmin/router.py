"""Root sysadmin router — aggregates all sub-routers."""
from fastapi import APIRouter

from app.sysadmin.auth.router import router as auth_router
from app.sysadmin.stats.router import router as stats_router
from app.sysadmin.tenants.router import router as tenants_router
from app.sysadmin.costs.router import router as costs_router
from app.sysadmin.audit.router import router as audit_router

sysadmin_router = APIRouter(prefix="/sysadmin")

sysadmin_router.include_router(auth_router)
sysadmin_router.include_router(stats_router)
sysadmin_router.include_router(tenants_router)
sysadmin_router.include_router(costs_router)
sysadmin_router.include_router(audit_router)
