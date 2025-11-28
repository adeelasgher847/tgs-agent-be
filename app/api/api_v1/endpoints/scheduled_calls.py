"""
Scheduled Calls API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from sqlalchemy.orm import Session
import uuid
from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.scheduled_call import (
    ScheduledCallResponse,
    ScheduledCallList,
    ScheduledCallUpdate,
    CSVUploadResponse
)
from app.services.scheduled_call_service import ScheduledCallService
from app.utils.response import create_success_response
from app.schemas.base import SuccessResponse

router = APIRouter()

scheduled_call_service = ScheduledCallService()


@router.post("/upload", response_model=SuccessResponse[CSVUploadResponse])
async def upload_scheduled_calls_csv(
    file: UploadFile = File(..., description="CSV file with scheduled calls"),
    timezone: str = Query("UTC", description="Default timezone for scheduled_time if not specified in CSV"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Upload CSV file to create scheduled calls.
    
    CSV format should have the following columns:
    - phone_number: Phone number to call (required)
    - agent_id: UUID of the agent (required)
    - scheduled_time: Scheduled time in format YYYY-MM-DD HH:MM:SS or ISO format (required)
    - timezone: Timezone string (optional, defaults to query parameter or UTC)
    
    Example CSV:
    phone_number,agent_id,scheduled_time,timezone
    +1234567890,550e8400-e29b-41d4-a716-446655440000,2024-01-15 14:30:00,America/New_York
    +0987654321,550e8400-e29b-41d4-a716-446655440000,2024-01-15 16:00:00,Europe/London
    
    The scheduled_time will be converted from the specified timezone to UTC before saving.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parse CSV and create scheduled calls
        result = scheduled_call_service.parse_csv_and_create_calls(
            db=db,
            tenant_id=user.current_tenant_id,
            user_id=user.id,
            csv_content=csv_content,
            user_timezone=timezone
        )
        
        return create_success_response(
            result,
            f"Processed {result.total_rows} rows: {result.successful_rows} successful, {result.failed_rows} failed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


@router.get("/", response_model=SuccessResponse[ScheduledCallList])
async def get_scheduled_calls(
    skip: int = Query(0, ge=0, description="Number of records to skip (for pagination)"),
    limit: int = Query(50, ge=1, le=50, description="Maximum number of records to return (max 50)"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get all pending calls based on current UTC time with pagination.
    
    Returns only calls with status 'pending' that are scheduled for the future.
    Maximum 50 records per request. Use skip and limit for pagination.
    
    Tenant ID and User ID are automatically extracted from the access token.
    Only returns calls for the authenticated user's tenant and user.
    
    This endpoint is designed for workflow automation to fetch and initiate calls.
    
    Example:
    - First page: skip=0, limit=50
    - Second page: skip=50, limit=50
    - Third page: skip=100, limit=50
    """
    try:
        # Get tenant_id and user_id from access token (via require_tenant dependency)
        tenant_id = user.current_tenant_id
        user_id = user.id
        
        # Get pending calls with pagination
        calls, total = scheduled_call_service.get_pending_calls(
            db=db,
            tenant_id=tenant_id,
            user_id=user_id,
            skip=skip,
            limit=limit
        )
        
        call_responses = [
            ScheduledCallResponse(
                id=call.id,
                tenant_id=call.tenant_id,
                user_id=call.user_id,
                phone_number=call.phone_number,
                agent_id=call.agent_id,
                scheduled_time_utc=call.scheduled_time_utc,
                status=call.status,
                created_at=call.created_at,
                updated_at=call.updated_at
            )
            for call in calls
        ]
        
        return create_success_response(
            ScheduledCallList(
                calls=call_responses, 
                total=total,
                skip=skip,
                limit=limit
            ),
            f"Retrieved {len(call_responses)} pending calls (total: {total})"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scheduled calls: {str(e)}")


@router.patch("/{call_id}", response_model=SuccessResponse[ScheduledCallResponse])
async def update_scheduled_call_status(
    call_id: uuid.UUID,
    status_update: ScheduledCallUpdate,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Update the status of a scheduled call.
    
    Valid status values:
    - pending: Call is pending
    - scheduled: Call has been scheduled
    - failed: Call failed
    - completed: Call completed successfully
    
    Only the owner of the call (same tenant and user) can update the status.
    """
    try:
        # Get tenant_id and user_id from access token
        tenant_id = user.current_tenant_id
        user_id = user.id
        
        # Update the scheduled call status
        updated_call = scheduled_call_service.update_scheduled_call_status(
            db=db,
            call_id=call_id,
            new_status=status_update.status,
            tenant_id=tenant_id,
            user_id=user_id
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

