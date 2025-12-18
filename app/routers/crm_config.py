"""
CRM Configuration API endpoints
Supports Monday.com, ClickUp, Jira, and Trello
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid

from app.api.deps import get_db, require_tenant, require_owner
from app.models.user import User
from app.schemas.crm_config import (
    TenantCRMConfigCreate,
    TenantCRMConfigUpdate,
    TenantCRMConfigOut,
)
from app.services.crm_config_service import CRMConfigService
from app.utils.response import create_success_response
from app.schemas.base import SuccessResponse

router = APIRouter()

crm_config_service = CRMConfigService()


@router.post("", response_model=SuccessResponse[TenantCRMConfigOut])
async def create_crm_config(
    crm_config_data: TenantCRMConfigCreate,
    user: User = Depends(require_owner),
    db: Session = Depends(get_db)
):
    """
    Create a new global CRM configuration (Owner only).
    
    Supports all CRMs:
    - **Monday.com**: Requires `api_key` and optional `additional_config.workspace_id`
    - **ClickUp**: Requires `api_key` and optional `additional_config.space_id`, `folder_id`
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
        
        response_data = TenantCRMConfigOut(
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


@router.put("/{crm_config_id}", response_model=SuccessResponse[TenantCRMConfigOut])
async def update_crm_config(
    crm_config_id: str,
    update_data: TenantCRMConfigUpdate,
    user: User = Depends(require_owner),
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
        
        response_data = TenantCRMConfigOut(
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
    user: User = Depends(require_owner),
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


@router.get("", response_model=SuccessResponse[list[TenantCRMConfigOut]])
async def get_all_crm_configs(
    user: User = Depends(require_tenant),
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
                TenantCRMConfigOut(
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


