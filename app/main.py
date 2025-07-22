from fastapi import FastAPI
from app.api.api_v1.api import api_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Tenant SaaS Voice Agent Backend!"}

app.include_router(api_router, prefix="/api/v1") 