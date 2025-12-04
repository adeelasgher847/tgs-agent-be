"""
Scheduled Calls API endpoints with Monday.com integration (per-tenant boards)
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.scheduled_call import CSVUploadResponse, BoardInfoResponse, DeleteBoardItemsResponse
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
    Upload CSV file to create scheduled calls in Monday.com board.
    
    **CSV Format (3 columns only):**
    ```
    phone_number,agent_id,call_time_utc
    ```
    
    **Required Columns:**
    - `phone_number`: Phone number to call (e.g., +1234567890)
    - `agent_id`: UUID of the agent (must exist in your tenant)
    - `call_time_utc`: Scheduled time in UTC - ISO format or YYYY-MM-DD HH:MM:SS
    
    **Note:** `tenant_id` and `user_id` are automatically taken from your logged-in session.
    
    **Example CSV:**
    ```csv
    phone_number,agent_id,call_time_utc
    +1234567890,550e8400-e29b-41d4-a716-446655440000,2024-12-02T14:30:00Z
    +0987654321,550e8400-e29b-41d4-a716-446655440000,2024-12-02 16:00:00
    ```
    
    **Flow:**
    1. Backend parses CSV and validates data
    2. Creates items in the tenant's dedicated Monday.com board (status: "Pending")
    3. n8n cron (every 1 min) detects new items
    4. n8n waits until call_time_utc
    5. n8n calls backend `/voice/call/initiate`
    6. n8n updates Monday.com status ("Called" or "Failed")
    
    **Data storage:** CSV rows live only in Monday.com. The backend stores one board
    record per tenant so we can re-use the same board on future uploads.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        result = await scheduled_call_service.parse_csv_and_send_to_monday(
            db=db,
            tenant_id=user.current_tenant_id,
            user_id=user.id,
            csv_content=csv_content
        )

        message = (
            f"Processed {result.total_rows} rows: {result.successful_rows} added to Monday.com, "
            f"{result.failed_rows} failed. Board URL: {result.board_url}"
        )

        return create_success_response(result, message)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


@router.get("/board", response_model=SuccessResponse[BoardInfoResponse])
async def get_board_url(user: User = Depends(require_tenant), db: Session = Depends(get_db)):
    """
    Retrieve the Monday.com board URL for the current tenant.
    """
    board_record = scheduled_call_service.get_board_for_tenant(db, user.current_tenant_id)
    if not board_record:
        raise HTTPException(status_code=404, detail="No scheduled calls board found for this tenant")

    data = BoardInfoResponse(
        board_id=board_record.monday_board_id,
        board_url=board_record.monday_board_url,
    )
    return create_success_response(data, "Scheduled calls board retrieved")


@router.delete("/board/items", response_model=SuccessResponse[DeleteBoardItemsResponse])
async def clear_board_items(user: User = Depends(require_tenant), db: Session = Depends(get_db)):
    """
    Remove all items from the tenant's Monday.com scheduled calls board, keeping the columns intact.
    """
    board_record, deleted = scheduled_call_service.clear_board_items(db, user.current_tenant_id)
    data = DeleteBoardItemsResponse(
        items_deleted=deleted,
        board_id=board_record.monday_board_id,
        board_url=board_record.monday_board_url,
    )
    return create_success_response(data, f"Deleted {deleted} item(s) from the board")
