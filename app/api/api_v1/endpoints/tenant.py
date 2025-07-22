from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.tenant import TenantCreate, TenantOut
from app.models.tenant import Tenant
from app.api.deps import get_db

router = APIRouter()

@router.post("/", response_model=TenantOut)
def create_tenant(tenant_in: TenantCreate, db: Session = Depends(get_db)):
    tenant = db.query(Tenant).filter(Tenant.name == tenant_in.name).first()
    if tenant:
        raise HTTPException(status_code=400, detail="Tenant already exists")
    db_tenant = Tenant(name=tenant_in.name, schema_name=tenant_in.schema_name)
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    return db_tenant 