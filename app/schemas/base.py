from pydantic import BaseModel
from typing import Generic, TypeVar, Optional
from fastapi import status

T = TypeVar('T')

class BaseResponse(BaseModel, Generic[T]):
    data: T
    message: str = ""
    status_code: int = status.HTTP_200_OK

class SuccessResponse(BaseResponse[T]):
    pass

class ErrorResponse(BaseModel):
    message: str
    status_code: int = status.HTTP_400_BAD_REQUEST
    error: Optional[str] = None
