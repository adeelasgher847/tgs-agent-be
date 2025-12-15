from pydantic import BaseModel, Field
from typing import List

class CSVUploadResponse(BaseModel):
    total_rows: int
    successful_rows: int
    failed_rows: int
    errors: List[str] = Field(default_factory=list)
    board_id: str
    board_url: str
    batch_id: str  # Unique batch ID for this CSV upload


class BoardInfoResponse(BaseModel):
    board_id: str
    board_url: str


class DeleteBoardItemsResponse(BaseModel):
    items_deleted: int
    board_id: str
    board_url: str


class PendingCountResponse(BaseModel):
    board_id: str
    board_url: str
    tenant_id: str
    pending_count: int
    total_items: int


class SingleCallRequest(BaseModel):
    phone_number: str = Field(..., description="Phone number to call (e.g., +1234567890)")
    agent_id: str = Field(..., description="Agent ID (UUID)")
    call_time_utc: str = Field(..., description="Scheduled time in UTC - ISO format or YYYY-MM-DD HH:MM:SS")


class SingleCallResponse(BaseModel):
    monday_item_id: str
    board_id: str
    board_url: str
    phone_number: str
    agent_id: str
    call_time_utc: str
    batch_id: str  # Unique batch ID for this single call
    message: str