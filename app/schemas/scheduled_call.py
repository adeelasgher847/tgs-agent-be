from pydantic import BaseModel, Field
from typing import List

class CSVUploadResponse(BaseModel):
    total_rows: int
    successful_rows: int
    failed_rows: int
    errors: List[str] = Field(default_factory=list)
    board_id: str
    board_url: str


class BoardInfoResponse(BaseModel):
    board_id: str
    board_url: str


class DeleteBoardItemsResponse(BaseModel):
    items_deleted: int
    board_id: str
    board_url: str
