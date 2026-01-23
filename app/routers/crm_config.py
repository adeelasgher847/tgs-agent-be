"""
CRM Configuration API endpoints
Supports Monday.com, ClickUp, Jira, and Trello
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid

from app.api.deps import get_db, require_tenant, require_owner, require_admin_or_owner, get_current_user_jwt, require_active_subscription
from app.models.user import User
from app.schemas.crm_config import (
    CRMConfigCreate,
    CRMConfigUpdate,
    CRMConfigOut,
)
from app.services.crm_config_service import CRMConfigService
from app.utils.response import create_success_response
from app.schemas.base import SuccessResponse
from app.core.config import settings
from app.models.tenant import Tenant
from app.models.plan import Plan
import stripe

router = APIRouter()

crm_config_service = CRMConfigService()


@router.post("", response_model=SuccessResponse[CRMConfigOut])
async def create_crm_config(
    crm_config_data: CRMConfigCreate,
    user: User = Depends(require_active_subscription),
    owner_user: User = Depends(require_owner),
    db: Session = Depends(get_db)
):
    """
    Create a new global CRM configuration (Owner only).
    
    Supports all CRMs:
    - **Monday.com**: Requires `api_key` and optional `additional_config.workspace_id`
    - **ClickUp**: Requires OAuth setup. First create config with `additional_config.client_id`, `additional_config.client_secret`, and `additional_config.redirect_uri`. Then use `/api/v1/auth/clickup/authorize` to get authorization URL, and after user authorizes, access token will be stored automatically. Optional: `additional_config.space_id`, `folder_id`
    - **Jira**: Requires `api_key`, `additional_config.email`, `additional_config.server_url`
    - **Trello**: Requires `api_key`, `additional_config.api_token`
    
    **Access:** Only Owner role can create CRM configurations.
    **Note:** CRM configs are global - all users can select any configured CRM.
    
    **Example Request (Monday.com):**
    ```json
    {
        "crm_type": "monday",
        "api_key": "your_monday_api_key",
        "additional_config": {
            "workspace_id": "optional_workspace_id"
        }
    }
    ```
    
    **Example Request (ClickUp OAuth):**
    ```json
    {
        "crm_type": "clickup",
        "api_key": null,
        "additional_config": {
            "client_id": "your_clickup_client_id",
            "client_secret": "your_clickup_client_secret",
            "redirect_uri": "https://yourdomain.com/api/v1/auth/clickup/callback",
            "space_id": "optional_space_id"
        }
    }
    ```
    Note: `api_key` can be `null` for ClickUp OAuth. After creating this config, call `/api/v1/auth/clickup/authorize` to get authorization URL. Access token will be automatically stored after OAuth callback.
    
    **Example Request (Trello):**
    ```json
    {
        "crm_type": "trello",
        "api_key": "your_trello_api_key",
        "additional_config": {
            "api_token": "your_trello_api_token"
        }
    }
    ```
    """
    # Validate CRM type
    valid_crm_types = ["monday", "clickup", "jira", "trello"]
    if crm_config_data.crm_type.lower() not in valid_crm_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid CRM type. Must be one of: {', '.join(valid_crm_types)}"
        )
    
    try:
        crm_config = crm_config_service.create_crm_config(
            db=db,
            crm_config_data=crm_config_data,
            created_by=user.id
        )
        
        # Parse additional_config for response
        additional_config_dict = None
        if crm_config.additional_config:
            import json
            additional_config_dict = json.loads(crm_config.additional_config)
            # Don't expose encrypted tokens in response
            if "api_token" in additional_config_dict:
                additional_config_dict["api_token"] = "***encrypted***"
        
        response_data = CRMConfigOut(
            id=crm_config.id,
            crm_type=crm_config.crm_type,
            container_id=crm_config.container_id,
            container_url=crm_config.container_url,
            additional_config=additional_config_dict,
            created_at=crm_config.created_at.isoformat() if crm_config.created_at else "",
            updated_at=crm_config.updated_at.isoformat() if crm_config.updated_at else None,
        )
        
        return create_success_response(
            response_data,
            f"CRM configuration for {crm_config.crm_type} created successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create CRM configuration: {str(e)}"
        )


@router.put("/{crm_config_id}", response_model=SuccessResponse[CRMConfigOut])
async def update_crm_config(
    crm_config_id: str,
    update_data: CRMConfigUpdate,
    user: User = Depends(require_active_subscription),
    owner_user: User = Depends(require_owner),
    db: Session = Depends(get_db)
):
    """
    Update an existing global CRM configuration (Owner only).
    
    **Access:** Only Owner role can update CRM configurations.
    
    **Note:** Only provided fields will be updated. Omitted fields remain unchanged.
    
    **Example Request:**
    ```json
    {
        "api_key": "new_api_key",
        "container_id": "new_container_id",
        "additional_config": {
            "workspace_id": "new_workspace_id"
        }
    }
    ```
    """
    try:
        crm_config_uuid = uuid.UUID(crm_config_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid CRM config ID format"
        )
    
    # Verify CRM config exists
    crm_config = crm_config_service.get_crm_config_by_id(db, crm_config_uuid)
    if not crm_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CRM configuration not found"
        )
    
    try:
        updated_config = crm_config_service.update_crm_config(
            db=db,
            crm_config_id=crm_config_uuid,
            update_data=update_data
        )
        
        # Parse additional_config for response
        additional_config_dict = None
        if updated_config.additional_config:
            import json
            additional_config_dict = json.loads(updated_config.additional_config)
            # Don't expose encrypted tokens in response
            if "api_token" in additional_config_dict:
                additional_config_dict["api_token"] = "***encrypted***"
        
        response_data = CRMConfigOut(
            id=updated_config.id,
            crm_type=updated_config.crm_type,
            container_id=updated_config.container_id,
            container_url=updated_config.container_url,
            additional_config=additional_config_dict,
            created_at=updated_config.created_at.isoformat() if updated_config.created_at else "",
            updated_at=updated_config.updated_at.isoformat() if updated_config.updated_at else None,
        )
        
        return create_success_response(
            response_data,
            f"CRM configuration for {updated_config.crm_type} updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update CRM configuration: {str(e)}"
        )


@router.delete("/{crm_config_id}", response_model=SuccessResponse[dict])
async def delete_crm_config(
    crm_config_id: str,
    user: User = Depends(require_active_subscription),
    owner_user: User = Depends(require_owner),
    db: Session = Depends(get_db)
):
    """
    Delete a global CRM configuration (Owner only).
    
    **Access:** Only Owner role can delete CRM configurations.
    
    **Warning:** This will permanently delete the CRM configuration.
    Any scheduled calls using this CRM config will need to be reconfigured.
    """
    try:
        crm_config_uuid = uuid.UUID(crm_config_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid CRM config ID format"
        )
    
    # Verify CRM config exists
    crm_config = crm_config_service.get_crm_config_by_id(db, crm_config_uuid)
    if not crm_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CRM configuration not found"
        )
    
    try:
        crm_config_service.delete_crm_config(db, crm_config_uuid)
        
        return create_success_response(
            {"deleted": True, "crm_config_id": crm_config_id},
            f"CRM configuration for {crm_config.crm_type} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete CRM configuration: {str(e)}"
        )


@router.get("", response_model=SuccessResponse[list[CRMConfigOut]])
async def get_all_crm_configs(
    user: User = Depends(require_active_subscription),
    db: Session = Depends(get_db)
):
    """
    Get all global CRM configurations.
    
    Returns a list of all configured CRMs (Monday.com, ClickUp, Jira, Trello).
    All users can see all configured CRMs.
    """
    try:
        crm_configs = crm_config_service.get_all_crm_configs(db)
        
        response_list = []
        for crm_config in crm_configs:
            # Parse additional_config for response
            additional_config_dict = None
            if crm_config.additional_config:
                import json
                additional_config_dict = json.loads(crm_config.additional_config)
                # Don't expose encrypted tokens in response
                if "api_token" in additional_config_dict:
                    additional_config_dict["api_token"] = "***encrypted***"
            
            response_list.append(
                CRMConfigOut(
                    id=crm_config.id,
                    crm_type=crm_config.crm_type,
                    container_id=crm_config.container_id,
                    container_url=crm_config.container_url,
                    additional_config=additional_config_dict,
                    created_at=crm_config.created_at.isoformat() if crm_config.created_at else "",
                    updated_at=crm_config.updated_at.isoformat() if crm_config.updated_at else None,
                )
            )
        
        return create_success_response(
            response_list,
            f"Retrieved {len(response_list)} CRM configuration(s)"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve CRM configurations: {str(e)}"
        )


@router.post("/start-checkout")
def start_checkout_session(
    stripe_price_id: str,
    current_user: User = Depends(get_current_user_jwt),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db)
):
    """
    Start a one-time Stripe checkout for a plan purchase.
    Credits will be granted after verification: $1 = 10 credits.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant selected"
        )
    tenant_id = str(current_user.current_tenant_id)
    tenant = db.query(Tenant).filter(Tenant.id == current_user.current_tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    # Create Stripe customer if not exists
    if not tenant.stripe_customer_id:
        from app.services.stripe_service import StripeService
        stripe_customer_id = StripeService.create_customer(
            tenant=tenant,
            email=current_user.email,
            user=current_user
        )
        tenant.stripe_customer_id = stripe_customer_id
        db.commit()
    else:
        stripe_customer_id = tenant.stripe_customer_id
    # Create checkout session directly with Stripe (one-time payment)
    
    stripe.api_key = settings.STRIPE_SECRET_KEY
    
    success_url = f"{settings.FRONTEND_URL}/payment/success?tenant_id={tenant_id}&session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{settings.FRONTEND_URL}/payment/cancel?tenant_id={tenant_id}"
    
    try:
        # Lookup plan to compute amount and embed metadata
        plan = db.query(Plan).filter(Plan.stripe_price_id == stripe_price_id).first()
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found for the given stripe_price_id"
            )
        raw_amount = int(plan.price_monthly or 0)
        amount_cents = raw_amount if raw_amount >= 50 else raw_amount * 100
        if amount_cents < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plan amount is not configured"
            )
        amount_dollars = amount_cents / 100.0

        checkout_session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            success_url=success_url,
            cancel_url=cancel_url,
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": f"Plan Purchase - {plan.display_name}"},
                    "unit_amount": amount_cents
                },
                "quantity": 1
            }],
            metadata={
                "tenant_id": tenant_id,
                "purchase_type": "plan_purchase",
                "plan_id": str(plan.id),
                "amount": str(amount_dollars)
            }
        )

        return create_success_response({
            "session_id": checkout_session.id,
            "url": checkout_session.url
        }, "Checkout session created successfully")
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


