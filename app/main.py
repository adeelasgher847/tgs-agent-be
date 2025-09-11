from fastapi import FastAPI, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse
from sqlalchemy.orm import Session

from app.api.api_v1.api import api_router
from app.routers.health import router as health_router
from app.routers.voice_processing import router as voice_processing_router
from app.schemas.base import SuccessResponse
from app.utils.response import create_success_response

app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # "http://localhost:5173",  # Your frontend dev server
        # "http://localhost:3000",  # Alternative frontend port
        # "http://127.0.0.1:5173",  # Alternative localhost
        # "http://127.0.0.1:3000",  # Alternative localhost
        # "http://192.168.0.121:5173",  # Your IP with frontend port
        "*"  # Allow all origins (for development only)
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/", response_model=SuccessResponse[dict])
def read_root():
    return create_success_response(
        {"message": "Welcome to the Multi-Tenant SaaS Voice Agent Backend!"},
        "API is running successfully"
    )
    
app.include_router(api_router, prefix="/api/v1")
app.include_router(health_router)