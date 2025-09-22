"""
Call Logs Router
Handles call logs management and retrieval for dashboard display
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import csv
import io
from datetime import datetime

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.call_log import (
    CallLogResponse, CallLogList, CallLogFilters, CallLogStats, CallLogDashboardResponse
)
from app.services.call_log_service import call_log_service
from app.utils.response import create_success_response

router = APIRouter()

@router.get("/logs", response_model=CallLogList)
async def get_call_logs(
    # Pagination
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    
    # Sorting
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    
    # Filters
    call_type: Optional[str] = Query(None, description="Filter by call type (inbound, outbound, web)"),
    success_evaluation: Optional[str] = Query(None, description="Filter by success evaluation (success, fail)"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    transferred: Optional[bool] = Query(None, description="Filter by transferred status"),
    ended_reason: Optional[str] = Query(None, description="Filter by ended reason"),
    assistant_phone_number: Optional[str] = Query(None, description="Filter by assistant phone number"),
    customer_phone_number: Optional[str] = Query(None, description="Filter by customer phone number"),
    
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get call logs with filtering, sorting, and pagination
    
    This endpoint provides a dashboard-like interface for viewing call logs,
    similar to the Vapi dashboard shown in the image.
    """
    try:
        # Parse agent_id if provided
        agent_uuid = None
        if agent_id:
            try:
                agent_uuid = uuid.UUID(agent_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid agent ID format")
        
        # Create filters object
        filters = CallLogFilters(
            call_type=call_type,
            success_evaluation=success_evaluation,
            agent_id=agent_uuid,
            date_from=date_from,
            date_to=date_to,
            transferred=transferred,
            ended_reason=ended_reason,
            assistant_phone_number=assistant_phone_number,
            customer_phone_number=customer_phone_number
        )
        
        # Get call logs with filters
        logs, total, stats = call_log_service.get_call_logs_with_filters(
            db=db,
            tenant_id=user.current_tenant_id,
            filters=filters,
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return CallLogList(
            logs=logs,
            total=total,
            stats=stats,
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/{log_id}", response_model=CallLogResponse)
async def get_call_log(
    log_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get a specific call log by ID"""
    try:
        call_log = call_log_service.get_call_log_by_id(db, log_id)
        
        if not call_log:
            raise HTTPException(status_code=404, detail="Call log not found")
        
        # Check if the call log belongs to the user's tenant
        if call_log.tenant_id != user.current_tenant_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return CallLogResponse.from_orm(call_log)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/recent", response_model=List[CallLogDashboardResponse])
async def get_recent_call_logs(
    limit: int = Query(10, ge=1, le=50, description="Number of recent logs to return"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get recent call logs for dashboard overview"""
    try:
        logs = call_log_service.get_recent_call_logs(
            db=db,
            tenant_id=user.current_tenant_id,
            limit=limit
        )
        
        return logs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=CallLogStats)
async def get_call_log_stats(
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get call log statistics for dashboard"""
    try:
        filters = CallLogFilters(date_from=date_from, date_to=date_to)
        stats = call_log_service._calculate_call_log_stats(
            db=db,
            tenant_id=user.current_tenant_id,
            filters=filters
        )
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export")
async def export_call_logs(
    # Filters (same as get_call_logs)
    call_type: Optional[str] = Query(None, description="Filter by call type"),
    success_evaluation: Optional[str] = Query(None, description="Filter by success evaluation"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    date_from: Optional[datetime] = Query(None, description="Filter from date"),
    date_to: Optional[datetime] = Query(None, description="Filter to date"),
    transferred: Optional[bool] = Query(None, description="Filter by transferred status"),
    ended_reason: Optional[str] = Query(None, description="Filter by ended reason"),
    
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Export call logs to CSV format"""
    try:
        # Parse agent_id if provided
        agent_uuid = None
        if agent_id:
            try:
                agent_uuid = uuid.UUID(agent_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid agent ID format")
        
        # Create filters object
        filters = CallLogFilters(
            call_type=call_type,
            success_evaluation=success_evaluation,
            agent_id=agent_uuid,
            date_from=date_from,
            date_to=date_to,
            transferred=transferred,
            ended_reason=ended_reason
        )
        
        # Get export data
        export_data = call_log_service.export_call_logs(
            db=db,
            tenant_id=user.current_tenant_id,
            filters=filters
        )
        
        # Create CSV
        output = io.StringIO()
        if export_data:
            fieldnames = export_data[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_data)
        
        # Return CSV response
        csv_content = output.getvalue()
        output.close()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=call_logs.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quick-filters")
async def get_quick_filters(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Get quick filter counts for dashboard (like Vapi's quick filters)"""
    try:
        # Get basic stats for quick filters
        filters = CallLogFilters()
        stats = call_log_service._calculate_call_log_stats(
            db=db,
            tenant_id=user.current_tenant_id,
            filters=filters
        )
        
        return {
            "all_calls": stats.total_calls,
            "transferred": stats.transferred_calls,
            "successful": stats.successful_calls,
            "failed": stats.failed_calls,
            "note": "Quick filters show counts for currently loaded results only."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
