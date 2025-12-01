"""
Scheduled Calls API endpoints - Automation only (no DB storage)
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.scheduled_call import CSVUploadResponse
from app.services.scheduled_call_service import ScheduledCallService
from app.utils.response import create_success_response
from app.schemas.base import SuccessResponse

router = APIRouter()

scheduled_call_service = ScheduledCallService()


@router.post("", response_model=SuccessResponse[CSVUploadResponse])
async def upload_scheduled_calls_csv(
    file: UploadFile = File(..., description="CSV file with scheduled calls"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Upload CSV file to trigger n8n webhooks for scheduled calls automation.
    
    CSV format should have the following columns:
    - phone_number: Phone number to call (required)
    - agent_id: UUID of the agent (required)
    - call_time_utc: Scheduled time in UTC (required) - ISO format (YYYY-MM-DDTHH:MM:SSZ) or YYYY-MM-DD HH:MM:SS
    
    Example CSV:
    phone_number,agent_id,call_time_utc
    +1234567890,550e8400-e29b-41d4-a716-446655440000,2024-01-15T14:30:00Z
    +0987654321,550e8400-e29b-41d4-a716-446655440000,2024-01-15 16:00:00
    
    After parsing each row, a webhook is sent to n8n with:
    - schedule_id (auto-generated UUID), tenant_id, user_id, phone_number, agent_id, call_time_utc
    
    The call_time_utc should already be in UTC format.
    No data is stored in the database - only webhooks are sent to n8n for automation.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parse CSV and send webhooks to n8n (no DB storage)
        result = await scheduled_call_service.parse_csv_and_send_webhooks(
            db=db,
            tenant_id=user.current_tenant_id,
            user_id=user.id,
            csv_content=csv_content
        )
        
        return create_success_response(
            result,
            f"Processed {result.total_rows} rows: {result.successful_rows} successful, {result.failed_rows} failed. Webhooks sent to n8n."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")
