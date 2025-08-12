from typing import Any, Optional
from fastapi import status
from app.schemas.base import BaseResponse, SuccessResponse, ErrorResponse

def create_success_response(
    data: Any, 
    message: str = "Success", 
    status_code: int = status.HTTP_200_OK
) -> SuccessResponse:
    """Create a standardized success response"""
    return SuccessResponse(
        data=data,
        message=message,
        status_code=status_code
    )

def create_error_response(
    message: str, 
    status_code: int = status.HTTP_400_BAD_REQUEST,
    error: Optional[str] = None
) -> ErrorResponse:
    """Create a standardized error response"""
    return ErrorResponse(
        message=message,
        status_code=status_code,
        error=error
    )
