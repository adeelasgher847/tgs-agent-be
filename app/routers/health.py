from fastapi import APIRouter
from app.schemas.base import SuccessResponse
from app.utils.response import create_success_response

router = APIRouter()

@router.get("/health", response_model=SuccessResponse[dict])
def health_check():
    return create_success_response({"status": "ok"}, "Health check successful") 