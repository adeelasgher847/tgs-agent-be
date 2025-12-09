"""
Scheduled Calls API endpoints with Monday.com integration (per-user boards).
All tenants of a user share the same board, identified by tenant_id column in items.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_
import uuid
from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.models.agent import Agent
from app.schemas.scheduled_call import CSVUploadResponse, BoardInfoResponse, DeleteBoardItemsResponse
from app.services.scheduled_call_service import ScheduledCallService
from app.utils.response import create_success_response
from app.schemas.base import SuccessResponse

router = APIRouter()

scheduled_call_service = ScheduledCallService()


@router.post("", response_model=SuccessResponse[CSVUploadResponse])
async def upload_scheduled_calls_csv(
    file: UploadFile = File(..., description="CSV file with scheduled calls"),
    agent_id: str = Query(..., description="Agent ID to use for all calls in this CSV (required)"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Upload CSV file to create scheduled calls in Monday.com board.
    
    **CSV Format (2 columns only):**
    ```
    phone_number,call_time_utc
    ```
    
    **Required:**
    - Select agent before upload (agent_id query parameter - required)
    - CSV with phone_number and call_time_utc only
    
    **Required Columns:**
    - `phone_number`: Phone number to call (e.g., +1234567890)
    - `call_time_utc`: Scheduled time in UTC - ISO format or YYYY-MM-DD HH:MM:SS
    
    **Note:** `tenant_id` and `user_id` are automatically taken from your logged-in session.
    All calls in this CSV will use the selected agent.
    
    **Example CSV:**
    ```csv
    phone_number,call_time_utc
    +1234567890,2024-12-02T14:30:00Z
    +0987654321,2024-12-02T14:31:00Z
    +1234567892,2024-12-02T14:32:00Z
    ```
    
    **Flow:**
    1. Select agent from dropdown
    2. Upload CSV (2 columns: phone_number, call_time_utc)
    3. Backend parses CSV and validates data
    4. Creates items in the user's Monday.com board (status: "Pending", tenant_id stored in column)
    5. n8n cron (every 1 min) detects new items
    6. n8n waits until call_time_utc
    7. n8n calls backend `/voice/call/initiate`
    8. n8n updates Monday.com status ("Called" or "Failed")
    
    **Data storage:** CSV rows live only in Monday.com. The backend stores one board
    record per user (shared by all their tenants). Items are identified by tenant_id column.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Validate and verify agent_id (REQUIRED)
        try:
            agent_uuid = uuid.UUID(agent_id)
            # Verify agent exists and belongs to tenant
            agent = db.query(Agent).filter(
                and_(
                    Agent.id == agent_uuid,
                    Agent.tenant_id == user.current_tenant_id,
                    Agent.is_deleted == False
                )
            ).first()
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found or doesn't belong to tenant")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        result = await scheduled_call_service.parse_csv_and_send_to_monday(
            db=db,
            tenant_id=user.current_tenant_id,
            user_id=user.id,
            csv_content=csv_content,
            default_agent_id=agent_uuid  # Pass selected agent (required)
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
    Retrieve the Monday.com board URL for the current user.
    All tenants of this user share the same board.
    """
    board_record = scheduled_call_service.get_board_for_user(db, user.id)
    if not board_record:
        raise HTTPException(status_code=404, detail="No scheduled calls board found for this user")

    data = BoardInfoResponse(
        board_id=board_record.monday_board_id,
        board_url=board_record.monday_board_url,
    )
    return create_success_response(data, "Scheduled calls board retrieved")


@router.delete("/board/items", response_model=SuccessResponse[DeleteBoardItemsResponse])
async def clear_board_items(user: User = Depends(require_tenant), db: Session = Depends(get_db)):
    """
    Remove all items belonging to the current tenant from the user's Monday.com board.
    Only items with matching tenant_id are deleted, keeping other tenants' items intact.
    """
    board_record, deleted = scheduled_call_service.clear_board_items(
        db, 
        user.id,  # user_id
        user.current_tenant_id  # tenant_id for filtering
    )
    data = DeleteBoardItemsResponse(
        items_deleted=deleted,
        board_id=board_record.monday_board_id,
        board_url=board_record.monday_board_url,
    )
    return create_success_response(data, f"Deleted {deleted} item(s) for current tenant from the board")
