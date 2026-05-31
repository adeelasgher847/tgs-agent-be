from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.schemas.plan import PlanOut, PlanCreate, PlanUpdate
from app.schemas.base import SuccessResponse
from app.models.plan import Plan
from app.api.deps import get_db, get_current_user_jwt, require_admin_or_owner
from app.utils.response import create_success_response
from typing import List, Optional
import uuid

router = APIRouter()

@router.get("/", response_model=SuccessResponse[List[PlanOut]])
def get_plans(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_jwt)
):
    """Get all available plans (authenticated)"""
    plans = db.query(Plan).filter(Plan.is_active == True).all()
    return create_success_response(plans, "Plans retrieved successfully")

@router.get("/public", response_model=SuccessResponse[List[PlanOut]])
def get_plans_public(
    db: Session = Depends(get_db)
):
    """Get all available plans (public - no authentication required)"""
    plans = db.query(Plan).filter(Plan.is_active == True).all()
    return create_success_response(plans, "Plans retrieved successfully")

@router.get("/popular", response_model=SuccessResponse[List[PlanOut]])
def get_popular_plans(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_jwt)
):
    """Get popular plans"""
    plans = db.query(Plan).filter(
        Plan.is_active == True,
        Plan.is_popular == True
    ).all()
    return create_success_response(plans, "Popular plans retrieved successfully")

@router.get("/{plan_id}", response_model=SuccessResponse[PlanOut])
def get_plan(
    plan_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_jwt)
):
    """Get a specific plan by ID"""
    plan = db.query(Plan).filter(Plan.id == plan_id).first()
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan not found"
        )
    return create_success_response(plan, "Plan retrieved successfully")

@router.get("/name/{plan_name}", response_model=SuccessResponse[PlanOut])
def get_plan_by_name(
    plan_name: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user_jwt)
):
    """Get a specific plan by name (e.g., 'free', 'starter', 'pro')"""
    plan = db.query(Plan).filter(Plan.name == plan_name, Plan.is_active == True).first()
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan '{plan_name}' not found"
        )
    return create_success_response(plan, "Plan retrieved successfully")

@router.get("/public/name/{plan_name}", response_model=SuccessResponse[PlanOut])
def get_plan_by_name_public(
    plan_name: str,
    db: Session = Depends(get_db)
):
    """Get a specific plan by name (public - no authentication required)"""
    plan = db.query(Plan).filter(Plan.name == plan_name, Plan.is_active == True).first()
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan '{plan_name}' not found"
        )
    return create_success_response(plan, "Plan retrieved successfully")

@router.post("/", response_model=SuccessResponse[PlanOut])
def create_plan(
    plan_data: PlanCreate,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin_or_owner)
):
    """Create a new plan (admin or owner only)"""
    # Check if plan with same name already exists
    existing_plan = db.query(Plan).filter(Plan.name == plan_data.name).first()
    if existing_plan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Plan with this name already exists"
        )
    
    plan = Plan(**plan_data.model_dump())
    db.add(plan)
    db.commit()
    db.refresh(plan)
    
    return create_success_response(plan, "Plan created successfully")
