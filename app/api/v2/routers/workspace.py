from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.models.branding_configs import BrandingConfig
from app.models.pricing_configs import PricingConfig
from app.models.usage_record import UsageRecord
from app.schemas.workspace import (
    BrandingConfigUpsert,
    BrandingConfigOut,
    PricingConfigUpsert,
    PricingConfigOut,
    WorkspaceUsageOut,
)


v2_router = APIRouter()

@v2_router.get("/branding", response_model=BrandingConfigOut)
def get_branding_config(
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get the current branding configuration for the workspace."""
    config = db.query(BrandingConfig).filter(BrandingConfig.workspace_id == user.current_tenant_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Branding configuration not found")
    return config

@v2_router.put("/branding", response_model=BrandingConfigOut)
def upsert_branding_config(
    payload: BrandingConfigUpsert,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upsert the branding configuration for the workspace."""
    from sqlalchemy.dialects.postgresql import insert
    stmt = insert(BrandingConfig).values(
        workspace_id=user.current_tenant_id,
        logo_url=payload.logo_url,
        primary_colour=payload.primary_colour,
        display_name=payload.display_name,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=['workspace_id'],
        set_={
            'logo_url': stmt.excluded.logo_url,
            'primary_colour': stmt.excluded.primary_colour,
            'display_name': stmt.excluded.display_name,
        }
    )
    db.execute(stmt)
    db.commit()
    return db.query(BrandingConfig).filter(BrandingConfig.workspace_id == user.current_tenant_id).first()

@v2_router.get("/pricing", response_model=PricingConfigOut)
def get_pricing_config(
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get the current pricing configuration for the workspace."""
    from decimal import Decimal
    config = db.query(PricingConfig).filter(PricingConfig.workspace_id == user.current_tenant_id).first()
    
    if not config:
        per_minute_rate = Decimal("0.12")
        markup_percent = Decimal("0.00")
    else:
        per_minute_rate = config.per_minute_rate
        markup_percent = config.markup_percent
        
    effective_client_rate = Decimal(str(per_minute_rate)) * (Decimal("1") + Decimal(str(markup_percent)) / Decimal("100"))
    
    return PricingConfigOut(
        per_minute_rate=per_minute_rate,
        markup_percent=markup_percent,
        effective_client_rate=effective_client_rate
    )

@v2_router.put("/pricing", response_model=PricingConfigOut)
def upsert_pricing_config(
    payload: PricingConfigUpsert,
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upsert the pricing configuration for the workspace."""
    from decimal import Decimal
    from sqlalchemy.dialects.postgresql import insert
    stmt = insert(PricingConfig).values(
        workspace_id=user.current_tenant_id,
        per_minute_rate=payload.per_minute_rate,
        markup_percent=payload.markup_percent,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=['workspace_id'],
        set_={
            'per_minute_rate': stmt.excluded.per_minute_rate,
            'markup_percent': stmt.excluded.markup_percent,
        }
    )
    db.execute(stmt)
    db.commit()
    
    config = db.query(PricingConfig).filter(PricingConfig.workspace_id == user.current_tenant_id).first()
    effective_client_rate = Decimal(str(config.per_minute_rate)) * (Decimal("1") + Decimal(str(config.markup_percent)) / Decimal("100"))
    
    return PricingConfigOut(
        per_minute_rate=config.per_minute_rate,
        markup_percent=config.markup_percent,
        effective_client_rate=effective_client_rate
    )

@v2_router.get("/usage", response_model=WorkspaceUsageOut)
def get_workspace_usage(
    user=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get the usage statistics for the current billing cycle."""
    from sqlalchemy import func
    from decimal import Decimal
    
    usage_sum = db.query(func.sum(UsageRecord.billable_minutes)).filter(
        UsageRecord.workspace_id == user.current_tenant_id,
        UsageRecord.recorded_at >= func.date_trunc('month', func.now())
    ).scalar() or Decimal("0")
    
    minutes_used_this_cycle = Decimal(str(usage_sum))
    minutes_included = None
    
    overage_minutes = max(Decimal("0"), minutes_used_this_cycle - minutes_included)
    
    config = db.query(PricingConfig).filter(PricingConfig.workspace_id == user.current_tenant_id).first()
    if config:
        effective_rate = Decimal(str(config.per_minute_rate)) * (Decimal("1") + Decimal(str(config.markup_percent)) / Decimal("100"))
    else:
        effective_rate = Decimal("0.12")
        
    overage_cost = overage_minutes * effective_rate
    
    return WorkspaceUsageOut(
        minutes_used_this_cycle=minutes_used_this_cycle,
        minutes_included=minutes_included,
        overage_minutes=overage_minutes,
        overage_cost=overage_cost
    )
