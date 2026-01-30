from pydantic import BaseModel, Field
from typing import List, Optional

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
    item_id: str
    board_id: str
    board_url: str
    phone_number: str
    agent_id: str
    call_time_utc: str
    batch_id: str  # Unique batch ID for this single call
    crm_type: str  # CRM type: "monday" | "clickup" | "jira" | "trello"
    message: str


class SelectCrmConfigRequest(BaseModel):
    """Request body for linking user with one or more CRMs (multi-CRM support)."""
    crm_config_ids: List[str] = Field(..., min_length=1, description="List of CRM config IDs (UUIDs) to link with user")


class JiraBatchAnalysisRequest(BaseModel):
    """Request body for Jira batch analysis endpoint"""
    call_session_ids: List[str] = Field(..., description="List of call session IDs (UUIDs)")
    phone_numbers: List[str] = Field(default_factory=list, description="List of phone numbers (optional, fallback)")
    total_scheduled: int = Field(..., description="Total scheduled calls")
    item_ids: List[str] = Field(default_factory=list, description="Item IDs for CRM update (e.g., issue_keys for Jira)")
    container_id: Optional[str] = Field(None, description="Container ID (project_key for Jira, optional)")