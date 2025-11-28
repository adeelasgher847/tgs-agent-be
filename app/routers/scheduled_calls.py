"""
Scheduled Calls API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status, Request
from sqlalchemy.orm import Session
from typing import Optional
import uuid
from app.api.deps import get_db, require_tenant, get_optional_tenant_user
from app.models.user import User
from app.schemas.scheduled_call import (
    ScheduledCallResponse,
    ScheduledCallUpdate,
    CSVUploadResponse
)
from app.services.scheduled_call_service import ScheduledCallService
from app.utils.response import create_success_response
from app.utils.n8n_webhook_verification import verify_n8n_webhook_secret_async
from app.schemas.base import SuccessResponse

router = APIRouter()

scheduled_call_service = ScheduledCallService()


async def _upload_scheduled_calls_csv_internal(
    file: UploadFile,
    user: User,
    db: Session
):
    """Internal function to handle CSV upload logic"""
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    # Read file content
    content = await file.read()
    csv_content = content.decode('utf-8')
    
    # Parse CSV and create scheduled calls (async, triggers n8n webhooks)
    result = await scheduled_call_service.parse_csv_and_create_calls(
        db=db,
        tenant_id=user.current_tenant_id,
        user_id=user.id,
        csv_content=csv_content
    )
    
    return create_success_response(
        result,
        f"Processed {result.total_rows} rows: {result.successful_rows} successful, {result.failed_rows} failed"
    )


@router.post("", response_model=SuccessResponse[CSVUploadResponse])
async def upload_scheduled_calls_csv(
    file: UploadFile = File(..., description="CSV file with scheduled calls"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Upload CSV file to create scheduled calls and trigger n8n webhooks.
    
    CSV format should have the following columns:
    - phone_number: Phone number to call (required)
    - agent_id: UUID of the agent (required)
    - call_time_utc: Scheduled time in UTC (required) - ISO format (YYYY-MM-DDTHH:MM:SSZ) or YYYY-MM-DD HH:MM:SS
    - status: Status (optional, defaults to "pending")
    
    Example CSV:
    phone_number,agent_id,call_time_utc,status
    +1234567890,550e8400-e29b-41d4-a716-446655440000,2024-01-15T14:30:00Z,pending
    +0987654321,550e8400-e29b-41d4-a716-446655440000,2024-01-15 16:00:00,pending
    
    After creating each scheduled call, a webhook is sent to n8n with:
    - schedule_id, tenant_id, user_id, phone_number, agent_id, call_time_utc
    
    The call_time_utc should already be in UTC format.
    """
    try:
        return await _upload_scheduled_calls_csv_internal(file, user, db)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


@router.patch("/{id}", response_model=SuccessResponse[ScheduledCallResponse])
async def update_scheduled_call_status(
    id: uuid.UUID,
    status_update: ScheduledCallUpdate,
    request: Request,
    tenant_id: Optional[str] = Query(None, description="Tenant ID (required if using webhook secret)"),
    user_id: Optional[str] = Query(None, description="User ID (optional, for filtering)"),
    user: Optional[User] = Depends(get_optional_tenant_user),
    db: Session = Depends(get_db)
):
    """
    Update the status of a scheduled call.
    
    This endpoint is typically called by n8n after a call is completed or fails.
    
    Authentication: Either JWT token OR n8n webhook secret (X-N8N-Webhook-Secret header).
    If using webhook secret, provide tenant_id (and optionally user_id) as query parameters.
    
    Valid status values:
    - pending: Call is pending
    - scheduled: Call has been scheduled
    - failed: Call failed
    - completed: Call completed successfully
    
    Only the owner of the call (same tenant and user) can update the status.
    """
    try:
        # Verify authentication: either JWT token OR webhook secret
        is_webhook = await verify_n8n_webhook_secret_async(request)
        
        if is_webhook:
            # Webhook authentication - get tenant_id and user_id from query params
            if not tenant_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="tenant_id query parameter is required when using webhook secret"
                )
            try:
                tenant_uuid = uuid.UUID(tenant_id)
                user_uuid = uuid.UUID(user_id) if user_id else None
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid UUID format for tenant_id or user_id"
                )
            tenant_id_filter = tenant_uuid
            user_id_filter = user_uuid
        else:
            # JWT authentication - get from user token
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required: JWT token or n8n webhook secret"
                )
            tenant_id_filter = user.current_tenant_id
            user_id_filter = user.id
        
        # Update the scheduled call status
        updated_call = scheduled_call_service.update_scheduled_call_status(
            db=db,
            call_id=id,
            new_status=status_update.status,
            tenant_id=tenant_id_filter,
            user_id=user_id_filter
        )
        
        # Convert to response model
        call_response = ScheduledCallResponse(
            id=updated_call.id,
            tenant_id=updated_call.tenant_id,
            user_id=updated_call.user_id,
            phone_number=updated_call.phone_number,
            agent_id=updated_call.agent_id,
            scheduled_time_utc=updated_call.scheduled_time_utc,
            status=updated_call.status,
            created_at=updated_call.created_at,
            updated_at=updated_call.updated_at
        )
        
        return create_success_response(
            call_response,
            f"Scheduled call status updated to '{status_update.status}' successfully"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update scheduled call status: {str(e)}")

